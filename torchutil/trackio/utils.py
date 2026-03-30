from pathlib import Path
from typing import TYPE_CHECKING

def set_trackio_dir(trackio, path: Path):
    """Changes the location of TRACKIO_DIR in all submodules of torchutil's private instance of `trackio`"""

    if TYPE_CHECKING:
        import trackio
        import trackio.sqlite_storage
        import trackio.media.media
        import trackio.media.utils

    # patch all instances of TRACKIO_DIR and MEDIA_DIR
    # TODO figure out a better way?
    trackio.TRACKIO_DIR = path
    trackio.sqlite_storage.TRACKIO_DIR = path
    trackio.media.media.MEDIA_DIR = path
    trackio.media.utils.MEDIA_DIR = path
    trackio.utils.MEDIA_DIR = path
    trackio.sqlite_storage.MEDIA_DIR = path

    # Clear all context_vars
    trackio.context_vars.current_run.set(None)
    trackio.context_vars.current_project.set(None)
    trackio.context_vars.current_server.set(None)
    trackio.context_vars.current_space_id.set(None)