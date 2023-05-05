"""
A sanity check that can be run to verify that the first dimensions of the feature encodings correspond
to the lowest resolution grids, and that the last dimensions correspond to the highest resolution grids.
"""

import tinycudann as tcnn
import torch

device = torch.device("cuda")
samples = 1000

encoding = tcnn.Encoding(
    n_input_dims=3,
    encoding_config={
        "base_resolution": 16,
        "hash": "CoherentPrime",
        "interpolation": "Nearest",  # NEED NEAREST (ie: no interpolation) FOR THIS TEST TO WORK
        "log2_hashmap_size": 19,
        "n_features_per_level": 2,
        "n_levels": 5,
        "otype": "Grid",
        "per_level_scale": 2.0,
        "type": "Hash",
    },
).to(device)

with torch.no_grad():
    input = torch.stack(
        [torch.linspace(0, 1, samples), torch.zeros(samples) + 0.01, torch.zeros(samples) + 0.01], dim=-1
    ).to(device)
    outs = encoding(input)
    diff = outs[1:] != outs[:-1]  # Keeps track of whether the feature encodings change as our inputs change
    print("Number of changes along each dimension", torch.sum(diff, dim=0))
