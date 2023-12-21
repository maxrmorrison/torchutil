import os
from typing import Callable, Iterable, List, Optional

import tqdm


###############################################################################
# TQDM iterator
###############################################################################


def iterator(
    iterable: Iterable,
    message: Optional[str] = None,
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

    Returns
        Monitored iterable
    """
    return tqdm.tqdm(
        iterable,
        desc=message,
        dynamic_ncols=True,
        initial=initial,
        total=len(iterable) if total is None else total)


def multiprocess_iterator(
    process: Callable,
    iterable: Iterable,
    message: Optional[str] = None,
    initial: int = 0,
    total: Optional[int] = None,
    num_workers: int = os.cpu_count(),
    worker_chunk_size: Optional[int] = None
) -> List:
    """Create a multiprocess tqdm iterator

    Arguments
        process
            The function that each multiprocess worker will call
        iterable
            Items to iterate over
        message
            Static message to display
        initial
            Position to display corresponding to index zero of iterable
        total
            Length of the iterable; defaults to len(iterable)
        num_workers
            Multiprocessing pool size; defaults to number of logical CPU cores
        worker_chunk_size
            Number of items sent to each multiprocessing worker

    Returns
        Return values of calling process on each item, in original order
    """
    # Get total number of items
    total = len(iterable) if total is None else total

    # Number of workers should not exceed number of items
    num_workers = min(num_workers, total)

    # Limit chunk size to prevent OOM
    if worker_chunk_size is None:
        worker_chunk_size = \
            1 if num_workers == total else min(32, total // num_workers)

    # Multiprocess monitor
    return tqdm.contrib.concurrent.process_map(
        process,
        iterable,
        desc=message,
        dynamic_ncols=True,
        initial=initial,
        total=total,
        max_workers=num_workers,
        chunksize=worker_chunk_size)
