import os
import requests
import tarfile
import tempfile
import zipfile
from typing import Callable, Optional, Union
import tqdm


###############################################################################
# Constants
###############################################################################


HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/112.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}


###############################################################################
# Progress bar
###############################################################################


class DownloadProgressBar(tqdm.tqdm):
    """Progress bar with defaults for byte-scaled download progress"""

    def __init__(self, *args, **kwargs):
        kwargs.setdefault('unit', 'B')
        kwargs.setdefault('unit_scale', True)
        super().__init__(*args, **kwargs)


class _ProgressStream:
    """File-like proxy that advances a progress bar as bytes are read"""

    def __init__(self, stream, progress):
        self.stream = stream
        self.progress = progress

    def read(self, size=-1):
        chunk = self.stream.read(size)
        self.progress.update(len(chunk))
        return chunk

    def __getattr__(self, name):
        return getattr(self.stream, name)


def _progress_bar(progress_bar_class: Callable, total: Optional[int]):
    """Instantiate a progress bar and verify it supports manual updates"""
    progress = progress_bar_class(total=total)
    if not callable(getattr(progress, 'update', None)):
        raise TypeError(
            f'progress_bar_class {progress_bar_class} produced {type(progress)}, '
            'which has no callable update method; download progress bars are '
            'advanced manually and must support update(n)')
    return progress


###############################################################################
# Data download utilities
###############################################################################


def file(
    url: str,
    path: Union[str, bytes, os.PathLike],
    use_headers: bool = False,
    timeout: Optional[float] = None,
    progress_bar_class: Callable = DownloadProgressBar
):
    """Download file from url

    Arguments
        url - The URL to download
        path - The location to save results
        use_headers - Use headers to imitate a browser
        timeout - Initial connection and subsequent read timeout in seconds;
            not a limit on total download time. None waits indefinitely
        progress_bar_class - Called with the total size to create the progress
            bar; must produce an object with an `update` method
    """
    headers = HEADERS if use_headers else None
    with requests.get(
        url,
        stream=True,
        headers=headers,
        timeout=timeout
    ) as rstream:
        rstream.raise_for_status()
        total = int(rstream.headers.get('content-length', 0)) or None
        with open(path, 'wb') as fstream, _progress_bar(
            progress_bar_class,
            total
        ) as progress:
            for chunk in rstream.iter_content(chunk_size=128):
                fstream.write(chunk)
                progress.update(len(chunk))


def tarbz2(
    url: str,
    path: Union[str, bytes, os.PathLike],
    use_headers: bool = False,
    timeout: Optional[float] = None,
    progress_bar_class: Callable = DownloadProgressBar
):
    """Download and extract tar bz2 file to location

    Arguments
        url - The URL to download
        path - The location to save results
        use_headers - Use headers to imitate a browser
        timeout - Initial connection and subsequent read timeout in seconds;
            not a limit on total download time. None waits indefinitely
        progress_bar_class - Called with the total size to create the progress
            bar; must produce an object with an `update` method
    """
    headers = HEADERS if use_headers else None
    with requests.get(
        url,
        stream=True,
        headers=headers,
        timeout=timeout
    ) as rstream:
        rstream.raise_for_status()
        total = int(rstream.headers.get('content-length', 0)) or None
        with _progress_bar(progress_bar_class, total) as progress:
            stream = _ProgressStream(rstream.raw, progress)
            with tarfile.open(fileobj=stream, mode='r|bz2') as tstream:
                tstream.extractall(path)


def targz(
    url: str,
    path: Union[str, bytes, os.PathLike],
    use_headers: bool = False,
    timeout: Optional[float] = None,
    progress_bar_class: Callable = DownloadProgressBar
):
    """Download and extract tar gz file to location

    Arguments
        url - The URL to download
        path - The location to save results
        use_headers - Use headers to imitate a browser
        timeout - Initial connection and subsequent read timeout in seconds;
            not a limit on total download time. None waits indefinitely
        progress_bar_class - Called with the total size to create the progress
            bar; must produce an object with an `update` method
    """
    headers = HEADERS if use_headers else None
    with requests.get(
        url,
        stream=True,
        headers=headers,
        timeout=timeout
    ) as rstream:
        rstream.raise_for_status()
        total = int(rstream.headers.get('content-length', 0)) or None
        with _progress_bar(progress_bar_class, total) as progress:
            stream = _ProgressStream(rstream.raw, progress)
            with tarfile.open(fileobj=stream, mode='r|gz') as tstream:
                tstream.extractall(path)


def zip(
    url: str,
    path: Union[str, bytes, os.PathLike],
    use_headers: bool = False,
    timeout: Optional[float] = None,
    progress_bar_class: Callable = DownloadProgressBar
):
    """Download and extract zip file to location

    Arguments
        url - The URL to download
        path - The location to save results
        use_headers - Use headers to imitate a browser
        timeout - Initial connection and subsequent read timeout in seconds;
            not a limit on total download time. None waits indefinitely
        progress_bar_class - Called with the total size to create the progress
            bar; must produce an object with an `update` method
    """
    with tempfile.NamedTemporaryFile() as tmp:
        file(
            url,
            tmp.name,
            use_headers=use_headers,
            timeout=timeout,
            progress_bar_class=progress_bar_class)
        with zipfile.ZipFile(tmp, 'r') as zfile:
            zfile.extractall(path)
