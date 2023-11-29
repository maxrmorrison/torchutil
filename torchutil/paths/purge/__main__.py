import argparse
from pathlib import Path

import torchutil


###############################################################################
# Remove files and directories
###############################################################################


def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(
        description='Remove files and directories')
    parser.add_argument(
        '--globs',
        nargs='+',
        help='Globs matching paths to delete')
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
        '--force',
        action='store_true',
        help='Skip user confirmation of deletion')
    return parser.parse_args()


if __name__ == '__main__':
    torchutil.paths.purge(**vars(parse_args()))
