import os
from pathlib import Path
from typing import List, Optional, Union

import torchutil


###############################################################################
# Constants
###############################################################################


# Available units of memory
UNITS = ['B', 'KB', 'MB', 'GB', 'TB']


###############################################################################
# Measure dataset sizes
###############################################################################


def measure(
    globs: Optional[List[Union[str, List[str]]]] = None,
    roots: Optional[
        List[
            Union[
                Union[str, bytes, os.PathLike],
                List[Union[str, bytes, os.PathLike]]
            ]
        ]
    ] = None,
    recursive: bool = False,
    unit='B'
) -> Union[int, float]:
    """Measure data usage of files and directories

    Arguments
        globs
            Globs matching paths to measure
        roots
            Directories to apply glob searches; current directory by default
        recursive
            Apply globs to all subdirectories of root directories
        unit
            Unit of memory utilization (bytes to terabytes); default bytes
    """
    # Argument handling
    if not isinstance(globs, list):
        globs = [globs]
    if not isinstance(roots, list):
        roots = [roots]
    globs = ['*' if glob is None else glob for glob in globs]
    roots = [Path() if root is None else Path(root) for root in roots]

    # Get paths to delete
    paths = []
    for root in roots:
        root = Path(root)
        for glob in globs:
            paths.extend(root.rglob(glob) if recursive else root.glob(glob))

    # Measure paths
    total = 0
    for path in torchutil.iterator(paths, 'Measuring paths'):
        total += Path(path).stat().st_size

    # Convert to desired unit of measurement
    return total if unit == 'B' else total / (1024 ** UNITS.index(unit))
