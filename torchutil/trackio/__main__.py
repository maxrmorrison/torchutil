import sys
import argparse
from pathlib import Path
import torchutil.trackio
from typing import TYPE_CHECKING

import sqlite3

def run():
    # Parse only your arg, collect everything else
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--directory', required=True, type=Path)
    known, remaining = parser.parse_known_args()

    directory = vars(known)['directory']

    trackio = torchutil.trackio.trackio(directory)

    # Rewrite sys.argv so trackio sees a normal invocation
    sys.argv = ['trackio'] + remaining

    main = trackio.cli.main

    sys.exit(main())

if __name__ == '__main__':
    run()