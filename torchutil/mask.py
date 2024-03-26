import torch


def from_lengths(lengths: torch.Tensor) -> torch.Tensor:
    """Create boolean mask from sequence lengths

    Arguments
        lengths
            The integer-type sequence lengths

    Returns
        mask
            The boolean-type sequence mask
    """
    return lengths.unsqueeze(1) > torch.arange(
        lengths.max(),
        dtype=lengths.dtype,
        device=lengths.device
    ).unsqueeze(0)
