import os
import shutil
from pathlib import Path
from typing import List, Optional, Union

import torchutil


###############################################################################
# Purge
###############################################################################


def purge(
    globs: Union[str, List[str]],
    roots: Optional[
        Union[
            Union[str, bytes, os.PathLike],
            List[Union[str, bytes, os.PathLike]]
        ]] = None,
    recursive: bool = False,
    force: bool = False
) -> None:
    """Remove all files and directories within directory matching glob

    Arguments
        globs
            Globs matching paths to delete
        roots
            Directories to apply glob searches; current directory by default
        recursive
            Apply globs to all subdirectories of root directories
        force
            Skip user confirmation of deletion
    """
    # Argument handling
    roots = Path() if roots is None else Path(roots)
    if isinstance(globs, str):
        globs = [globs]
    if not isinstance(roots, list):
        roots = [roots]

    # Get paths to delete
    paths = []
    for root in roots:
        root = Path(root)
        for glob in globs:
            paths.extend(root.rglob(glob) if recursive else root.glob(glob))

    # Maybe prompt user
    if not force:
        files = len(path for path in paths if path.is_file())
        dirs = len(paths) - files
        if input(
            f'Found {files} files and {dirs} directories to delete within '
            f'{len(roots)} root directories. Continue? [y/N]:'
        ).lower() != 'y':
            raise RuntimeError('Purge aborted by user')

    # Delete paths
    for path in torchutil.iterator(paths, 'Deleting paths'):
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
