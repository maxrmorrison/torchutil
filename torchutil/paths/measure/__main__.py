import argparse
from pathlib import Path

import torchutil

###############################################################################
# Purge datasets
###############################################################################


def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description='Measure dataset disk usage')
    parser.add_argument(
        '--directories',
        nargs='+',
        type=Path,
        help='The datasets to measure')
    parser.add_argument(
        '--extensions',
        nargs='+',
        help='Which cached features to measure')
    return parser.parse_args()


torchutil.paths.measure.features_in_directories(**vars(parse_args()))
