import contextlib
import time
from typing import Optional


###############################################################################
# Timer context
###############################################################################


@contextlib.contextmanager
def context(name: Optional[str]):
    """Wrapper to handle context changes of global timer

    Arguments
        name - Name of the timer to add time to
    """
    # Initialize singleton timer
    if not hasattr(context, 'timer'):
        context.timer = Timer()

    # Allow nested timers
    previous = context.timer.name
    context.timer.name = name

    # Run timing
    with context.timer:
        yield

    # Restore state
    context.timer.name = previous


def reset():
    """Clear timer state"""
    try:
        del context.timer
    except AttributeError:
        pass


def results() -> dict:
    """Get timing results

    Returns
        Timing results: {name: elapsed_time} for all names
    """
    try:
        return context.timer()
    except AttributeError:
        return {}


###############################################################################
# Utilities
###############################################################################


class Timer:
    """Context manager timer"""

    def __init__(self):
        self.reset()

    def __call__(self):
        """Retrieve timer results"""
        results = {name: sum(times) for name, times in self.history.items()}
        results['total'] = self.total
        return results

    def __enter__(self):
        """Start the timer"""
        self.stack.append(time.time())

    def __exit__(self, *_):
        """Stop the timer"""
        # Get time elapsed
        elapsed = time.time() - self.stack.pop()

        # Add to timer history
        if self.name not in self.history:
            self.history[self.name] = [elapsed]
        else:
            self.history[self.name].append(elapsed)

        # Update total
        if not self.stack:
            self.total += elapsed

    def reset(self):
        """Reset the timer"""
        self.name = None
        self.history = {}
        self.stack = []
        self.total = 0.
