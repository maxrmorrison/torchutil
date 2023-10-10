import contextlib
import time


###############################################################################
# Timer context
###############################################################################


@contextlib.contextmanager
def context(name):
    """Wrapper to handle context changes of global timer"""
    if not hasattr(context, 'timer'):
        context.timer = Timer()

    # TODO - support nested contexts
    context.timer.name = name
    with context.timer:
        yield
    context.timer.name = None


def results():
    """Get timing results"""
    if not hasattr(context, 'timer'):
        return {}
    return context.timer()


###############################################################################
# Utilities
###############################################################################


class Timer:
    """Context manager timer"""

    def __init__(self):
        self.reset()

    def __call__(self):
        """Retrieve timer results"""
        return {name: sum(times) for name, times in self.history.items()}

    def __enter__(self):
        """Start the timer"""
        self.start = time.time()

    def __exit__(self, *_):
        """Stop the timer"""
        elapsed = time.time() - self.start

        # Add to timer history
        if self.name not in self.history:
            self.history[self.name] = [elapsed]
        else:
            self.history[self.name].append(elapsed)

    def reset(self):
        """Reset the timer"""
        self.history = {}
        self.start = 0.
        self.name = None
