import os
from typing import Callable, Dict, List, Optional, Tuple, Union

import torch


###############################################################################
# Checkpointing
###############################################################################


def load(
    file: Union[str, bytes, os.PathLike],
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    map_location: str = 'cpu') -> Tuple[
        torch.nn.Module,
        Union[None, torch.optim.Optimizer],
        Dict
    ]:
    """Load model checkpoint

    Arguments
        file - The checkpoint file
        model - The PyTorch model
        optimizer - Optional PyTorch optimizer for training
        map_location - The device to load the checkpoint on

    Returns
        model - The model with restored weights
        optimizer - Optional optimizer with restored parameters
        state - Additional values that the user defined during save
    """
    checkpoint = torch.load(file, map_location=map_location)

    # Restore model
    model.load_state_dict(checkpoint['model'])
    del checkpoint['model']

    # Restore optimizer
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer'])
    del checkpoint['optimizer']

    return model, optimizer, checkpoint


def save(
    file: Union[str, bytes, os.PathLike],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    accelerator=None,
    **kwargs):
    """Save training checkpoint to disk

    Arguments
        file - The checkpoint file
        model - The PyTorch model
        optimizer - The PyTorch optimizer
        accelerator - HuggingFace Accelerator for device management
        kwargs - Additional values to save
    """
    if accelerator is None:
        save_fn = torch.save
    else:
        save_fn = accelerator.save
    checkpoint = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict()}
    save_fn(checkpoint | kwargs, file)


###############################################################################
# Utilities
###############################################################################


def highest_number(
    files: List[Union[str, bytes, os.PathLike]]
) -> Union[str, bytes, os.PathLike]:
    """Default for determining latest path; assumes filename is steps

    Arguments
        files - The checkpoint files

    Returns
        The filename containing the largest number
    """
    files.sort(key=lambda path: int(''.join(filter(str.isdigit, path.stem))))
    return files[-1]


def latest_path(
        directory: Union[str, bytes, os.PathLike],
        regex: str = '*.pt',
        latest_fn: Callable = highest_number,
    ) -> Union[str, bytes, os.PathLike]:
    """Retrieve the path to the most recent checkpoint in a directory

    Arguments
        directory - The directory to search for checkpoint files
        regex - The regular expression matching checkpoints
        latest_fn - Takes a list of checkpoint paths and returns the latest
                    Default assumes checkpoint names are training step count.
    Returns
        The latest checkpoint in directory according to latest_fn
    """
    # Retrieve checkpoint filenames
    files = list(directory.glob(regex))

    # If no matching checkpoint files, no training has occurred
    if not files:
        return

    # Retrieve latest checkpoint
    return latest_fn(files)
