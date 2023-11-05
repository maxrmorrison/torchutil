import os
import shutil
from pathlib import Path
from typing import Optional, Union

import torchutil


###############################################################################
# Purge
###############################################################################


def purge(
    glob: str,
    root: Optional[Union[str, bytes, os.PathLike]] = None,
    recursive: bool = False
) -> None:
    """Remove all files and directories within directory matching glob

    Arguments
        glob
            Glob matching files to delete
        root
            Directory to apply glob search; current directory by default
        recursive
            Apply glob to all subdirectories of root
    """
    root = Path() if root is None else Path(root)
    paths = list(root.rglob(glob) if recursive else root.glob(glob))
    for path in torchutil.iterator(paths, 'Deleting paths'):
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
