from typing import Optional

import torch
from nerfstudio.model_components.losses import ray_samples_to_sdist
from nerfstudio.model_components.ray_samplers import SpacedSampler


def power_fn(x_s, lam):
    coeff = -1
    lam = torch.tensor(lam).to(x_s.device)
    base = (x_s / torch.abs(lam - 1)) + 1
    return coeff * (torch.pow(base, lam) - 1)


def inverse_power_fn(x_metric, lam):
    lam = torch.tensor(lam).to(x_metric.device)
    coeff = -1
    return (torch.pow(x_metric + 1, -lam) - 1) * torch.abs(lam - 1)


class PowerSampler(SpacedSampler):
    """Sampler according to the zipnerf's power function"""

    def __init__(
        self,
        num_samples: Optional[int] = None,
        train_stratified=True,
        single_jitter=False,
        lam=-1.5,
    ) -> None:
        super().__init__(
            num_samples=num_samples,
            spacing_fn=lambda x: power_fn(x, lam),
            spacing_fn_inv=lambda x: inverse_power_fn(x, lam),
            train_stratified=train_stratified,
            single_jitter=single_jitter,
        )


def blur_stepfun(x, y, r):
    # assert x.shape == y.shape, f"Shapes of x and y should be the same, got {x.shape} and {y.shape}"
    # assert x.shape[-1] != 1 and y.shape[-1] != 1, "Make sure that x, y are squeezed"
    xr, xr_idx = torch.sort(torch.cat([x - r, x + r], dim=-1))
    y1 = (
        torch.cat([y, torch.zeros_like(y[..., :1])], dim=-1) - torch.cat([torch.zeros_like(y[..., :1]), y], dim=-1)
    ) / (2 * r)
    y2 = torch.cat([y1, -y1], dim=-1).take_along_dim(xr_idx[..., :-1], dim=-1)
    yr = torch.cumsum((xr[..., 1:] - xr[..., :-1]) * torch.cumsum(y2, dim=-1), dim=-1).clamp_min(0)
    # yr = torch.cat([torch.zeros_like(yr[..., :1]), yr], dim=-1)
    # assert xr.shape == yr.shape, f"Shapes of xr and yr should be the same, got {xr.shape} and {yr.shape}"
    return xr, yr


def sorted_interp(x, xp, fp):
    """A TPU-friendly version of interp(), where xp and fp must be sorted."""

    # Identify the location in `xp` that corresponds to each `x`.
    # The final `True` index in `mask` is the start of the matching interval.
    mask = x[..., None, :] >= xp[..., :, None]

    def find_interval(x):
        # Grab the value where `mask` switches from True to False, and vice versa.
        # This approach takes advantage of the fact that `x` is sorted.
        x0 = torch.max(torch.where(mask, x[..., None], x[..., :1, None]), -2).values
        x1 = torch.min(torch.where(~mask, x[..., None], x[..., -1:, None]), -2).values
        return x0, x1

    fp0, fp1 = find_interval(fp)
    xp0, xp1 = find_interval(xp)

    offset = torch.clip(torch.nan_to_num((x - xp0) / (xp1 - xp0), 0), 0, 1)
    ret = fp0 + offset * (fp1 - fp0)
    return ret


def resample(t, tp, vp, use_avg=False):
    """Resample a step function defined by (tp, vp) into intervals t.
    Args:
      t: tensor with shape (..., n+1), the endpoints to resample into.
      tp: tensor with shape (..., m+1), the endpoints of the step function being
        resampled.
      vp: tensor with shape (..., m), the values of the step function being
        resampled.
      use_avg: bool, if False, return the sum of the step function for each
        interval in `t`. If True, return the average, weighted by the width of
        each interval in `t`.
      eps: float, a small value to prevent division by zero when use_avg=True.
    Returns:
      v: tensor with shape (..., n), the values of the resampled step function.
    """
    eps = torch.finfo(t.dtype).eps
    # eps = 1e-3

    if use_avg:
        wp = torch.diff(tp, dim=-1)
        v_numer = resample(t, tp, vp * wp, use_avg=False)
        v_denom = resample(t, tp, wp, use_avg=False)
        v = v_numer / v_denom.clamp_min(eps)
        return v

    acc = torch.cumsum(vp, dim=-1)
    acc0 = torch.cat([torch.zeros(acc.shape[:-1] + (1,), device=acc.device), acc], dim=-1)
    acc0_resampled = sorted_interp(t, tp, acc0)  # TODO
    v = torch.diff(acc0_resampled, dim=-1)
    return v


def zipnerf_proposal_loss(weights_list, ray_samples_list, blur_radiuses=torch.tensor([0.03, 0.003])):
    """Calculates the proposal loss in the ZipNerf paper."""
    blur_radiuses = blur_radiuses.to(weights_list[0].device)
    c = ray_samples_to_sdist(ray_samples_list[-1]).detach()
    w = weights_list[-1][..., 0].detach()
    w_normalize = w / (c[..., 1:] - c[..., :-1])
    # print(c.shape, w_normalize.shape)

    loss_interlevel = 0.0
    # Iterate over each proposal level's samples + weights, add the loss
    for ray_samples, weights, blur_radius in zip(ray_samples_list[:-1], weights_list[:-1], blur_radiuses):
        # Get out detached resampled nerf weights, that we can then
        sdist = ray_samples_to_sdist(ray_samples)
        cp = sdist  # (num_rays, num_samples + 1)
        wp = weights[..., 0]  # (num_rays, num_samples)
        with torch.no_grad():
            c_blurred, w_blurred = blur_stepfun(c, w_normalize, blur_radius)  # Blurred histogram

        w_s = resample(cp, c_blurred, w_blurred * (c_blurred[..., 1:] - c_blurred[..., :-1]))
        loss_interlevel += ((w_s - wp).clamp_min(0) ** 2 / (wp + 1e-5)).mean()
        # print(loss_interlevel)

        # assert (
        #     w_resampled.shape == weights.shape
        # ), f"Resampled weights {w_resampled.shape} should have same shape as proposal weights {weights.shape}"

        # loss_interlevel += torch.mean((1 / weights) * (torch.clamp(w_resampled - weights, min=0.0) ** 2))

    return loss_interlevel
