from pathlib import Path

import math

from typing import List, Optional

###############################################################################
# Measure dataset sizes
###############################################################################


def features_in_directories(directories: List[Path], extensions: Optional[List[str]] = ['.pt']):
    """Get total disk usage of features in directories based on file extensions"""

    total = 0
    for directory in directories:
        # Measure one directory
        subtotal = 0
        for extension in extensions:
            # Measure one feature in one directory
            feature_size = measure_glob(
                directory,
                f'**/*{extension}')
            print(f'\t{extension}: {size_to_string(feature_size)}')

            # Update directory total
            subtotal += feature_size

        # Update aggregate total
        print(f'{directory} is {size_to_string(subtotal)}')
        total += subtotal
    print(f'Total is {size_to_string(total)}')


###############################################################################
# Utilities
###############################################################################


def measure_glob(path, glob_string):
    """Get the size in bytes of all files matching glob"""
    return sum(file.stat().st_size for file in path.glob(glob_string))


def size_to_string(size_in_bytes):
    """Format size in gigabytes"""
    return f'{math.ceil(size_in_bytes / (1024 ** 3))} GB'
