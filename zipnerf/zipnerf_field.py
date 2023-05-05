# Copyright 2022 The Nerfstudio Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Field for compound nerf model, adds scene contraction and image embeddings to instant ngp
"""


from typing import Dict, Optional, Tuple

import numpy as np
import torch
from nerfstudio.cameras.rays import RaySamples
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.field_components.activations import trunc_exp
from nerfstudio.field_components.embedding import Embedding
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.field_components.mlp import MLP
from nerfstudio.field_components.spatial_distortions import (
    SceneContraction,
    SpatialDistortion,
)
from nerfstudio.fields.base_field import Field, shift_directions_for_tcnn
from torch import nn
from torch.nn.parameter import Parameter
from torchtyping import TensorType

from zipnerf.zipnerf_utils import erf

try:
    import tinycudann as tcnn
except ImportError:
    # tinycudann module doesn't exist
    pass


class ZipNeRFField(Field):
    """ZipNeRF Field

    Args:
        aabb: parameters of scene aabb bounds
        num_images: number of images in the dataset
        num_layers: number of hidden layers
        hidden_dim: dimension of hidden layers
        geo_feat_dim: output geo feat dimensions
        num_levels: number of levels of the hashmap for the base mlp
        max_res: maximum resolution of the hashmap for the base mlp
        log2_hashmap_size: size of the hashmap for the base mlp
        num_layers_color: number of hidden layers for color network
        num_layers_transient: number of hidden layers for transient network
        hidden_dim_color: dimension of hidden layers for color network
        hidden_dim_transient: dimension of hidden layers for transient network
        appearance_embedding_dim: dimension of appearance embedding
        use_pred_normals: whether to use predicted normals
        use_average_appearance_embedding: whether to use average appearance embedding or zeros for inference
        spatial_distortion: spatial distortion to apply to the scene
    """

    def __init__(
        self,
        aabb: TensorType,
        num_images: int,
        num_layers: int = 2,
        hidden_dim: int = 64,
        geo_feat_dim: int = 15,
        num_levels: int = 16,
        max_res: int = 2048,
        log2_hashmap_size: int = 19,
        num_layers_color: int = 3,
        hidden_dim_color: int = 64,
        appearance_embedding_dim: int = 32,
        use_average_appearance_embedding: bool = False,
        spatial_distortion: SpatialDistortion = None,
    ) -> None:
        super().__init__()

        self.register_buffer("aabb", aabb)
        self.geo_feat_dim = geo_feat_dim

        self.register_buffer("max_res", torch.tensor(max_res))
        self.register_buffer("num_levels", torch.tensor(num_levels))
        self.register_buffer("log2_hashmap_size", torch.tensor(log2_hashmap_size))

        self.spatial_distortion = spatial_distortion
        self.num_images = num_images
        self.appearance_embedding_dim = appearance_embedding_dim
        self.embedding_appearance = Embedding(self.num_images, self.appearance_embedding_dim)
        self.use_average_appearance_embedding = use_average_appearance_embedding

        base_res: int = 16
        self.base_resolution = base_res
        features_per_level: int = 4
        growth_factor = np.exp((np.log(max_res) - np.log(base_res)) / (num_levels - 1))
        self.growth_factor = growth_factor
        self.features_per_level = features_per_level

        self.direction_encoding = tcnn.Encoding(
            n_input_dims=3,
            encoding_config={
                "otype": "SphericalHarmonics",
                "degree": 4,
            },
        )

        self.hashgrid_encoding = tcnn.Encoding(
            n_input_dims=3,
            encoding_config={
                "otype": "HashGrid",
                "n_levels": num_levels,
                "n_features_per_level": features_per_level,
                "log2_hashmap_size": log2_hashmap_size,
                "base_resolution": base_res,
                "per_level_scale": growth_factor,
            },
        )

        self.density_backbone = tcnn.Network(
            n_input_dims=self.hashgrid_encoding.n_output_dims * 2,
            n_output_dims=1 + geo_feat_dim,
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": hidden_dim,
                "n_hidden_layers": num_layers - 1,
            },
        )

        self.mlp_head = tcnn.Network(
            n_input_dims=self.direction_encoding.n_output_dims + self.geo_feat_dim + self.appearance_embedding_dim,
            n_output_dims=3,
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "Sigmoid",
                "n_neurons": hidden_dim_color,
                "n_hidden_layers": num_layers_color - 1,
            },
        )

    def get_density(self, ray_samples: RaySamples) -> Tuple[TensorType, TensorType]:
        """Computes and returns the densities."""
        assert len(ray_samples.shape) == 2, "Not coded for multiple batch dimensions yet"
        variance_scale = 0.35  # Standard deviation of gaussians is scaled by .35 as per the paper
        n_multisamples = 1

        gaussians = ray_samples.frustums.get_positions()
        # [n_multisamples, B, 3]
        # positions = torch.distributions.MultivariateNormal(gaussians.mean, gaussians.cov).sample(
        #     torch.tensor([n_multisamples])
        # )
        positions = gaussians[None].expand(n_multisamples, -1, -1, -1)

        ts = torch.einsum("...ij,...ij->...i", positions, ray_samples.frustums.directions).unsqueeze(
            -1
        )  # [n_multisamples, B, n_raysamples, 1]
        rs = cone_radius = (
            torch.sqrt(ray_samples.frustums.pixel_area) / 1.7724538509055159
        )  # radii of the cone at the samples
        standard_deviations = variance_scale * rs * ts  # [n_multisamples, B, n_raysamples, 1]

        # positions = positions + (standard_deviations * torch.randn_like(standard_deviations))

        # Calculate the resolution corresponding to each element of our hashgrid, which goes from lowest res
        # to highest res
        levels = torch.tensor(
            [self.base_resolution * (self.growth_factor**i) for i in range(self.num_levels)], device=ts.device
        )
        nl = levels.repeat_interleave(self.features_per_level)

        multisample_weights = erf(1 / (2.82842712474619 * standard_deviations * nl))  # 1 / sqrt(8 * std**2 * nl**2)

        if self.spatial_distortion is not None:
            positions = self.spatial_distortion(positions)
            positions = (positions + 2.0) / 4.0
        else:
            positions = SceneBox.get_normalized_positions(positions, self.aabb)

        features = self.hashgrid_encoding(positions.view(-1, 3)).view(
            n_multisamples, ray_samples.frustums.shape[0], ray_samples.frustums.shape[1], -1
        )

        assert features.shape == multisample_weights.shape, f"{features.shape} != {multisample_weights.shape}"
        downweighted_features = features.mean(dim=0)  # (features * multisample_weights).mean(dim=0)
        featurized_weights = ((2 * multisample_weights.mean(dim=0)) - 1) * torch.sqrt(
            1 + (features.detach().mean(0) ** 2)
        )
        downweighted_features = torch.cat([downweighted_features, featurized_weights], dim=-1)
        _shape = downweighted_features.shape

        features = self.density_backbone(downweighted_features.view(-1, _shape[-1])).view(*_shape[:-1], -1)

        density_before_activation, base_mlp_out = torch.split(features, [1, self.geo_feat_dim], dim=-1)

        # Make sure the tcnn gets inputs between 0 and 1.
        selector = ((positions > 0.0) & (positions < 1.0)).all(dim=-1)
        # assert not torch.any(selector == False), f"Positions out of bounds: {positions[not selector]}"
        # positions = positions * selector[..., None]
        # self._sample_locations = positions
        # if not self._sample_locations.requires_grad:
        #     self._sample_locations.requires_grad = True
        # positions_flat = positions.view(-1, 3)
        # h = self.hashgrid_encoding(positions_flat).view(*ray_samples.frustums.shape, -1)
        # density_before_activation, base_mlp_out = torch.split(h, [1, self.geo_feat_dim], dim=-1)
        self._density_before_activation = density_before_activation

        # Rectifying the density with an exponential is much more stable than a ReLU or
        # softplus, because it enables high post-activation (float32) density outputs
        # from smaller internal (float16) parameters.
        density = trunc_exp(density_before_activation.to(positions))
        density = density * selector[..., None].all(dim=0)
        return density, base_mlp_out

    def get_outputs(
        self, ray_samples: RaySamples, density_embedding: Optional[TensorType] = None
    ) -> Dict[FieldHeadNames, TensorType]:
        assert density_embedding is not None
        outputs = {}
        if ray_samples.camera_indices is None:
            raise AttributeError("Camera indices are not provided.")
        camera_indices = ray_samples.camera_indices.squeeze()
        directions = shift_directions_for_tcnn(ray_samples.frustums.directions)
        directions_flat = directions.view(-1, 3)
        d = self.direction_encoding(directions_flat)

        outputs_shape = ray_samples.frustums.directions.shape[:-1]

        # appearance
        if self.training:
            embedded_appearance = self.embedding_appearance(camera_indices)
        else:
            if self.use_average_appearance_embedding:
                embedded_appearance = torch.ones(
                    (*directions.shape[:-1], self.appearance_embedding_dim), device=directions.device
                ) * self.embedding_appearance.mean(dim=0)
            else:
                embedded_appearance = torch.zeros(
                    (*directions.shape[:-1], self.appearance_embedding_dim), device=directions.device
                )

        h = torch.cat(
            [
                d,
                density_embedding.view(-1, self.geo_feat_dim),
                embedded_appearance.view(-1, self.appearance_embedding_dim),
            ],
            dim=-1,
        )
        rgb = self.mlp_head(h).view(*outputs_shape, -1).to(directions)
        outputs.update({FieldHeadNames.RGB: rgb})

        return outputs
