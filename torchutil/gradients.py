from typing import Dict

import torch


###############################################################################
# Gradient management
###############################################################################


def stats(model: torch.nn.Module) -> Dict[str, float]:
    """Get gradient statistics

    Arguments
        model
            The torch model

    Returns
        The L2 norm, maximum, and minimum gradients
    """
    total_norm = 0
    maximum_gradient = 0
    minimum_gradient = 0
    for p in model.parameters():
        if p.grad is not None:
            maximum_gradient = max(maximum_gradient, p.grad.data.max())
            minimum_gradient = min(minimum_gradient, p.grad.data.min())
            total_norm += p.grad.data.norm(2)
    total_norm = total_norm ** (1. / 2)
    return {
        'gradients/norm': total_norm,
        'gradients/max': maximum_gradient,
        'gradients/min': minimum_gradient}
