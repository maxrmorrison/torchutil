import os
from typing import Dict, Optional, Union

import accelerate
import torch
from torch.utils.tensorboard import SummaryWriter


###############################################################################
# Constants
###############################################################################


# Device managenent state
STATE = accelerate.state.PartialState()


###############################################################################
# Tensorboard logging
###############################################################################


@STATE.on_main_process
def update(
    directory: Union[str, bytes, os.PathLike],
    step: int,
    audio: Optional[Dict[str, torch.Tensor]] = None,
    sample_rate: Optional[int] = None,
    figures: Optional[Dict] = None,
    images: Optional[Dict[str, torch.Tensor]] = None,
    scalars: Optional[Dict[str, Union[float, int, torch.Tensor]]] = None):
    """Update Tensorboard

    Arguments
        directory - Directory to write Tensorboard files
        step - Training step
        audio - Optional dictionary of 2D audio tensors to monitor
        sample_rate - Audio sample rate; required if audio is not None
        figures - Optional dictionary of Matplotlib figures to monitor
        images - Optional dictionary of 3D image tensors to monitor
        scalars - Optional dictionary of scalars to monitor
    """
    if audio is not None:
        write_audio(directory, step, audio, sample_rate)
    if figures is not None:
        write_figures(directory, step, figures)
    if images is not None:
        write_images(directory, step, images)
    if scalars is not None:
        write_scalars(directory, step, scalars)


###############################################################################
# Writer
###############################################################################


@STATE.on_main_process
def writer(directory):
    """Get the writer object"""
    if not hasattr(writer, 'writer') or writer.directory != directory:
        writer.writer = SummaryWriter(log_dir=directory)
        writer.directory = directory
    return writer.writer


###############################################################################
# Utilities
###############################################################################


@STATE.on_main_process
def write_audio(directory, step, audio, sample_rate):
    """Write audio to Tensorboard"""
    for name, waveform in audio.items():
        writer(directory).add_audio(name, waveform, step, sample_rate)


@STATE.on_main_process
def write_figures(directory, step, figures):
    """Write figures to Tensorboard"""
    for name, figure in figures.items():
        writer(directory).add_figure(name, figure, step)


@STATE.on_main_process
def write_images(directory, step, images):
    """Write images to Tensorboard"""
    for name, image in images.items():
        writer(directory).add_image(name, image, step, dataformats='HCW')


@STATE.on_main_process
def write_scalars(directory, step, scalars):
    """Write scalars to Tensorboard"""
    for name, scalar in scalars.items():
        if isinstance(scalar, dict):
            for subname, subscalar in scalar.items():
                writer(directory).add_scalar(
                    name + '/' + subname,
                    subscalar,
                    step
                )
        else:
            writer(directory).add_scalar(name, scalar, step)
