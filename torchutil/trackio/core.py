import torchutil
import os
from typing import Union
from pathlib import Path

def trackio(directory: Union[str, bytes, os.PathLike]):
    import torchutil.trackio

    # import an instance of trackio that is separate from the one
    #  in sys.modules
    _trackio = torchutil.trackio.create_trackio_instance()

    # monkeypatch the TRACKIO_DIR and MEDIA_DIR variables on the
    #  private instance
    torchutil.trackio.set_trackio_dir(_trackio, Path(directory).absolute())

    # patch all sql and media related operations on the private instance
    #  so that separate runs have separate database files in separate directories
    torchutil.trackio.patch_sql(_trackio)

    return _trackio

