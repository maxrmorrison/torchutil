import trackio # this cannot be removed or things break
import sys
import importlib
from copy import copy
from typing import TYPE_CHECKING


def create_trackio_instance():
    # import a custom instance of the trackio module that will be used by torchutil

    # Create a lock to prevent multithreading issues
    # We store the lock directly in sys.modules since that's the
    #  shared resource that we care about (kinda genius no?)
    if f'torchutil.IMPORT_LOCK_TRACKIO' in sys.modules:
        raise RuntimeError(f"Two different threads tried to import trackio at the same time")
    sys.modules[f'torchutil.IMPORT_LOCK_TRACKIO'] = None

    # Temporarily remove configured modules from sys.modules to ensure
    # that other modules are configured properly
    to_restore = {}
    to_delete = []
    for module_name in copy(sys.modules).keys():
        if module_name.split('.')[0] == 'trackio':
            to_delete.append(module_name)
    for module_name in to_delete:
        to_restore[module_name] = sys.modules[module_name]
        del sys.modules[module_name]

    # Import the module
    _trackio = importlib.import_module('trackio')
    _trackio.cli = importlib.import_module('trackio.cli')

    # Revert sys.modules
    for module_name, module_object in to_restore.items():
        sys.modules[module_name] = module_object

    del sys.modules[f'torchutil.IMPORT_LOCK_TRACKIO']

    if TYPE_CHECKING:
        return trackio
    return _trackio