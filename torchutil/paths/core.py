import contextlib
import os
from typing import Union


###############################################################################
# Path management utilities
###############################################################################


@contextlib.contextmanager
def chdir(directory: Union[str, bytes, os.PathLike]) -> None:
    """Context manager for changing the current working directory

    Arguments
        directory
            The desired working directory
    """
    curr_dir = os.getcwd()
    try:
        os.chdir(directory)
        yield
    finally:
        os.chdir(curr_dir)
