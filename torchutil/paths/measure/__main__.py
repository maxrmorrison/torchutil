import argparse
import os
from pathlib import Path
from typing import List, Optional, Union

import torchutil


###############################################################################
# Measure files and directories
###############################################################################


def main(
    globs: Union[str, List[str]],
    roots: Optional[
        Union[
            Union[str, bytes, os.PathLike],
            List[Union[str, bytes, os.PathLike]]
        ]] = None,
    recursive: bool = False,
    unit='B'
) -> Union[int, float]:
    print(f'{torchutil.paths.measure(**locals())} {unit}')


def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(
        description='Measure data usage of files and directories')
    parser.add_argument(
        '--globs',
        required=True,
        nargs='+',
        help='Globs matching paths to measure')
    parser.add_argument(
        '--roots',
        type=Path,
        nargs='+',
        help='Directories to apply glob searches; '
              'current directory by default')
    parser.add_argument(
        '--recursive',
        action='store_true',
        help='Apply globs to all subdirectories of root directories')
    parser.add_argument(
        '--unit',
        choices=['B', 'KB', 'MB', 'GB', 'TB'],
        default='B',
        help='Unit of memory utilization (bytes to terabytes); default bytes')
    return parser.parse_args()


if __name__ == '__main__':
    main(**vars(parse_args()))
