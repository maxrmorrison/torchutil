try:
    import sqlite3
except ImportError as e:
    if 'CXXABI' in e.msg:
        raise ImportError("Not quite sure why this happens, " \
        "but try installing nvidia-cudnn-cuXX via conda/mamba")
from .private_instance import create_trackio_instance
from .utils import *
from .sql_patching import patch_sql
from .core import trackio