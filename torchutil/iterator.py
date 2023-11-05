from typing import Iterable, Optional

import tqdm


###############################################################################
# TQDM iterator
###############################################################################


def iterator(
    iterable: Iterable,
    message: Optional[str],
    initial: int = 0,
    total: Optional[int] = None
) -> Iterable:
    """Create a tqdm iterator

    Arguments
        iterable
            Items to iterate over
        message
            Static message to display
        initial
            Position to display corresponding to index zero of iterable
        total
            Length of the iterable; defaults to len(iterable)
    """
    return tqdm.tqdm(
        iterable,
        desc=message,
        dynamic_ncols=True,
        initial=initial,
        total=len(iterable) if total is None else total)
