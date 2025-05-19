import os
import requests
import tarfile
import tempfile
import zipfile
from typing import Union


###############################################################################
# Data download utilities
###############################################################################


def file(url: 'str', path: Union[str, bytes, os.PathLike], use_headers: bool = False):
    """Download file from url

    Arguments
        url - The URL to download
        path - The location to save results
        use_headers - Use headers to imitate a browser
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/112.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    } if use_headers else None
    with requests.get(url, stream=True, headers=headers) as rstream:
        rstream.raise_for_status()
        with open(path, 'wb') as fstream:
            for chunk in rstream.iter_content(chunk_size=128):
                fstream.write(chunk)


def tarbz2(url: 'str', path: Union[str, bytes, os.PathLike], use_headers: bool = False):
    """Download and extract tar bz2 file to location

    Arguments
        url - The URL to download
        path - The location to save results
        use_headers - Use headers to imitate a browser
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/112.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    } if use_headers else None
    with requests.get(url, stream=True, headers=headers) as rstream:
        rstream.raise_for_status()
        with tarfile.open(fileobj=rstream.raw, mode='r|bz2') as tstream:
            tstream.extractall(path)


def targz(url: 'str', path: Union[str, bytes, os.PathLike], use_headers: bool = False):
    """Download and extract tar gz file to location

    Arguments
        url - The URL to download
        path - The location to save results
        use_headers - Use headers to imitate a browser
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/112.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    } if use_headers else None
    with requests.get(url, stream=True, headers=headers) as rstream:
        rstream.raise_for_status()
        with tarfile.open(fileobj=rstream.raw, mode='r|gz') as tstream:
            tstream.extractall(path)


def zip(url: 'str', path: Union[str, bytes, os.PathLike], use_headers: bool = False):
    """Download and extract zip file to location

    Arguments
        url - The URL to download
        path - The location to save results
        use_headers - Use headers to imitate a browser
    """
    with tempfile.NamedTemporaryFile() as tmp:
        file(url, tmp.name, use_headers=use_headers)
        with zipfile.ZipFile(tmp, 'r') as zfile:
            zfile.extractall(path)
