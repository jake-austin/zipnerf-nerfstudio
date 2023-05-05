import torch


def erf(x):
    """ERF function approximation from zipnerf"""

    return 2 * torch.sign(x) * torch.sqrt(1 - torch.exp(-(4 / torch.pi) * (x**2)))
