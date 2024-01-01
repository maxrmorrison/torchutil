import contextlib
import functools

import torch


###############################################################################
# Inference utilities
###############################################################################


@contextlib.contextmanager
def context(model: torch.nn.Module, autocast: bool = True) -> None:
    """Inference-time handling of model training flag and optimizations

    Arguments
        model
            The torch model performing inference
        autocast
            Whether to use mixed precision
    """
    device_type = next(model.parameters()).device.type

    # Prepare model for evaluation
    model.eval()

    # Maybe use mixed precision
    if autocast:
        autocast_context = functools.partial(torch.autocast, device_type)
    else:
        autocast_context = contextlib.nullcontext

    # Turn off gradient computation; turn on mixed precision
    with torch.inference_mode(), autocast_context():
        yield

    # Prepare model for training
    model.train()
