import os
from typing import Dict, Optional, Union

import torch
from torch.utils.tensorboard import SummaryWriter


###############################################################################
# Tensorboard logging
###############################################################################


def update(
    directory: Union[str, bytes, os.PathLike],
    step: int,
    audio: Optional[Dict[str, torch.Tensor]] = None,
    sample_rate: Optional[int] = None,
    figures: Optional[Dict] = None,
    images: Optional[Dict[str, torch.Tensor]] = None,
    scalars: Optional[Dict[str, Union[float, int, torch.Tensor]]] = None,
    grouped_scalars: Optional[Dict[str, torch.Tensor]] = None,
    hyperparameters: Optional[Dict[str, Union[bool, str, float, int, None]]] = None
):
    """Update Tensorboard

    Arguments
        directory - Directory to write Tensorboard files
        step - Training step
        audio - Optional dictionary of 2D audio tensors to monitor
        sample_rate - Audio sample rate; required if audio is not None
        figures - Optional dictionary of Matplotlib figures to monitor
        images - Optional dictionary of 3D image tensors to monitor
        scalars - Optional dictionary of scalars to monitor
        grouped_scalars - Optional dictionary of dictionaries containing grouped scalars to monitor
        hyperparameters - Optional dictionary of hyperparameters to log
    """
    if audio is not None:
        write_audio(directory, step, audio, sample_rate)
    if figures is not None:
        write_figures(directory, step, figures)
    if images is not None:
        write_images(directory, step, images)
    if scalars is not None:
        write_scalars(directory, step, scalars)
    if grouped_scalars is not None:
        write_grouped_scalars(directory, step, grouped_scalars)
    if hyperparameters is not None:
        write_hyperparameters(directory, step, hyperparameters)


###############################################################################
# Writer
###############################################################################


def writer(directory):
    """Get the writer object"""
    if not hasattr(writer, 'writer') or writer.directory != directory:
        writer.writer = SummaryWriter(log_dir=directory)
        writer.directory = directory
    return writer.writer


###############################################################################
# Utilities
###############################################################################


def write_audio(directory, step, audio, sample_rate):
    """Write audio to Tensorboard"""
    for name, waveform in audio.items():
        writer(directory).add_audio(name, waveform, step, sample_rate)


def write_figures(directory, step, figures):
    """Write figures to Tensorboard"""
    for name, figure in figures.items():
        writer(directory).add_figure(name, figure, step)


def write_images(directory, step, images):
    """Write images to Tensorboard"""
    for name, image in images.items():
        writer(directory).add_image(name, image, step, dataformats='HCW')


def write_scalars(directory, step, scalars):
    """Write scalars to Tensorboard"""
    for name, scalar in scalars.items():
        writer(directory).add_scalar(name, scalar, step)


def write_grouped_scalars(directory, step, grouped_scalars):
    """Write grouped scalars to Tensorboard"""
    for name, scalars in grouped_scalars.items():
        tags = []
        for tag, value in scalars.items():
            full_tag = f'{name}/{tag}'
            w = writer(directory)
            w.add_scalar(full_tag, value, step)
            tags.append(full_tag)
        w.add_custom_scalars_multilinechart(tags)
        # writer(directory).add_scalars(name, scalars, step)

def write_hyperparameters(directory, step, hyperparameters):
    """Write hyperparameters to Tensorboard"""
    writer(directory).add_hparams(hyperparameters, global_step=step)
