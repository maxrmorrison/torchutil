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
    map_location: str = 'cpu'
) -> Tuple[
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
    **kwargs
) -> None:
    """Save training checkpoint to disk

    Arguments
        file - The checkpoint file
        model - The PyTorch model
        optimizer - The PyTorch optimizer
        kwargs - Additional values to save
    """
    checkpoint = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict()}
    torch.save(checkpoint | kwargs, file)


###############################################################################
# Search predicates
###############################################################################


def highest_score(
    files: List[Union[str, bytes, os.PathLike]],
    scores: List[float]
) -> Tuple[Union[str, bytes, os.PathLike], float]:
    """Default for determining best checkpoint

    Arguments
        files - The checkpoint files

    Returns
        best_file - The filename of the checkpoint with the best score
        best_score - The corresponding score
    """
    index = torch.argsort(torch.tensor(scores))[-1]
    return files[index], scores[index]


def largest_number_filename(
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


###############################################################################
# Search checkpoints
###############################################################################


def best_path(
    directory: Union[str, bytes, os.PathLike],
    glob: str = '*.pt',
    best_fn: Callable = highest_score
) -> Tuple[Union[str, bytes, os.PathLike], float]:
    """Retrieve the path to the best checkpoint

    Arguments
        directory - The directory to search for checkpoint files
        glob - The glob matching the checkpoints
        best_fn - Takes a list of checkpoint paths and returns the latest
                  Default assumes checkpoint names are training step count.

    Returns
        best_file - The filename of the checkpoint with the best score
        best_score - The corresponding score
    """
    # Retrieve checkpoint filenames
    files = list(directory.glob(glob))

    # If no matching checkpoint files, no training has occurred
    if not files:
        return

    # Get all scores
    scores = [torch.load(file, map_location='cpu')['score'] for file in files]
    return best_fn(files, scores)


def latest_path(
    directory: Union[str, bytes, os.PathLike],
    glob: str = '*.pt',
    latest_fn: Callable = largest_number_filename,
) -> Union[str, bytes, os.PathLike]:
    """Retrieve the path to the most recent checkpoint in a directory

    Arguments
        directory - The directory to search for checkpoint files
        glob - The glob matching the checkpoints
        latest_fn - Takes a list of checkpoint paths and returns the latest
                    Default assumes checkpoint names are training step count.
    Returns
        The latest checkpoint in directory according to latest_fn
    """
    # Retrieve checkpoint filenames
    files = list(directory.glob(glob))

    # If no matching checkpoint files, no training has occurred
    if not files:
        return

    # Retrieve latest checkpoint
    return latest_fn(files)
