import bdb
import contextlib
import os
import time


###############################################################################
# Send job notifications
###############################################################################


@contextlib.contextmanager
def on_finish(
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

    # Run user code
    try:
        yield
    except Exception as exception:

        # Ignore pdb exceptions
        if isinstance(exception, bdb.BdbQuit):
            return

        if notify_on_fail:

            # End time
            if track_time:
                end_time = time.time()

            # Report failure
            if track_time:
                message = (
                    f'Task "{description}" failed with '
                    'exception: {exception.__class__} in'
                    f'{round(end_time - start_time)} seconds')
            else:
                message = (
                    f'Task "{description}" finished in '
                    f'{round(end_time - start_time)} seconds')
            push(message)

        raise exception

    # End time
    if track_time:
        end_time = time.time()

    # Report success
    if track_time:
        message = (
            f'Task "{description}" finished in '
            f'{round(end_time - start_time)} seconds')
    else:
        message = f'Task "{description}" finished'
    push(message)


###############################################################################
# Utilities
###############################################################################


def push(message: str):
    """Send a push notification to all of the configured services"""
    service = os.getenv('PYTORCH_NOTIFICATION_URL', default=None)
    if service is not None:
        if not hasattr(push, 'messenger'):
            import apprise
            push.messenger = apprise.Apprise()
            push.messenger.add(service)
        push.messenger.notify(message)
