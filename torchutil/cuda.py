from typing import Dict

import torch


###############################################################################
# Constants
###############################################################################


# Available units of memory
UNITS = ['B', 'KB', 'MB', 'GB', 'TB']


###############################################################################
# CUDA utilities
###############################################################################


def utilization(device: torch.device, unit: str ='B') -> Dict[str, float]:
    """Get the current VRAM utilization of a specified device

    Arguments
        device
            The device to query for VRAM utilization
        unit
            Unit of memory utilization (bytes to terabytes); default bytes

    Returns
        Allocated and reserved VRAM utilization in the specified unit
    """
    index = device.index
    scale = (1024 ** UNITS.index(unit))
    return {
        f'cuda/maximum_allocated ({unit})':
            torch.cuda.max_memory_allocated(index) / scale,
        f'cuda/maximum_reserved ({unit})':
            torch.cuda.max_memory_reserved(index) / scale}
