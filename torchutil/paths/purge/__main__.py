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
        '--glob',
        required=True,
        help='Glob matching files to delete')
    parser.add_argument(
        '--root',
        type=Path,
        default=Path(),
        help='Directory to apply glob search; current directory by default')
    parser.add_argument(
        '--recursive',
        action='store_true',
        help='Apply glob to all subdirectories of root')
    return parser.parse_args()


if __name__ == '__main__':
    torchutil.paths.purge(**vars(parse_args()))
