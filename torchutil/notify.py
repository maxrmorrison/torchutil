import bdb
import contextlib
import os
import time
from typing import Callable

import apprise


###############################################################################
# Send job notifications
###############################################################################


@contextlib.contextmanager
def on_exit(
    description: str,
    track_time: bool = True,
    notify_on_fail: bool = True):
    """Context manager for sending job notifications

    Arguments
        description - The name of the job being run
        track_time - Whether to report time elapsed
        notify_on_fail - Whether to send a notification on failure
    """
    # Start time
    if track_time:
        start_time = time.time()

    try:

        # Run user code
        yield

    except Exception as exception:

        # Ignore pdb exceptions
        if isinstance(exception, bdb.BdbQuit):
            return

        if notify_on_fail:

            # End time
            elapsed = time.time() - start_time if track_time else None

            # Report failure
            push_failure(description, track_time, elapsed, exception)

        raise exception

    # End time
    elapsed = time.time() - start_time if track_time else None

    # Report success
    push_success(description, track_time, elapsed)


def on_return(
    description: str,
    track_time: bool = True,
    notify_on_fail: bool = True) -> Callable:
    """Decorator for sending job notifications

    Arguments
        description - The name of the job being run
        track_time - Whether to report time elapsed
        notify_on_fail - Whether to send a notification on failure
    """
    def wrapper(func: callable):
        def _wrapper(*args, **kwargs):
            # Start time
            if track_time:
                start_time = time.time()

            # Run callable
            try:
                func(*args, **kwargs)
            except Exception as exception:

                # Report failure; ignore pdb exceptions
                if notify_on_fail and not isinstance(exception, bdb.BdbQuit):

                    # End time
                    elapsed = time.time() - start_time if track_time else None

                    # Report failure
                    push_failure(description, track_time, elapsed, exception)

                raise exception

            # End time
            elapsed = time.time() - start_time if track_time else None

            # Report success
            push_success(description, track_time, elapsed)

        return _wrapper
    return wrapper


###############################################################################
# Utilities
###############################################################################


def push(message: str):
    """Send a push notification to all of the configured services"""
    service = os.getenv('PYTORCH_NOTIFICATION_URL', default=None)
    if service is not None:
        if not hasattr(push, 'messenger'):
            push.messenger = apprise.Apprise()
            push.messenger.add(service)
        push.messenger.notify(message)


def push_failure(description, track_time, elapsed, exception):
    """Send notification of task failure"""
    if track_time:
        message = (
            f'Task "{description}" failed with '
            f'exception: {exception.__class__} in '
            f'{elapsed:.2f} seconds')
    else:
        message = (
            f'Task "{description}" failed with '
            f'exception: {exception.__class__}')
    push(message)


def push_success(description, track_time, elapsed):
    """Send notification of successful task completion"""
    if track_time:
        message = (
            f'Task "{description}" finished in '
            f'{elapsed:.2f} seconds')
    else:
        message = f'Task "{description}" finished'
    push(message)
