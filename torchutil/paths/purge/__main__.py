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
        required=True,
        help='Globs matching files to delete')
    parser.add_argument(
        '--roots',
        type=Path,
        default=Path(),
        help='Directories to apply glob searches; '
              'current directory by default')
    parser.add_argument(
        '--recursive',
        action='store_true',
        help='Apply globs to all subdirectories of root directories')
    return parser.parse_args()


if __name__ == '__main__':
    torchutil.paths.purge(**vars(parse_args()))
