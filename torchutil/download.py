import os
import requests
import tarfile
import tempfile
import zipfile
from typing import Union


###############################################################################
# Data download utilities
###############################################################################


def file(url: 'str', path: Union[str, bytes, os.PathLike]):
    """Download file from url

    Arguments
        url - The URL to download
        path - The location to save results
    """
    with requests.get(url, stream=True) as rstream:
        rstream.raise_for_status()
        with open(path, 'wb') as fstream:
            for chunk in rstream.iter_content(chunk_size=128):
                fstream.write(chunk)


def tarbz2(url: 'str', path: Union[str, bytes, os.PathLike]):
    """Download and extract tar bz2 file to location

    Arguments
        url - The URL to download
        path - The location to save results
    """
    with requests.get(url, stream=True) as rstream:
        rstream.raise_for_status()
        with tarfile.open(fileobj=rstream.raw, mode='r|bz2') as tstream:
            tstream.extractall(path)


def targz(url: 'str', path: Union[str, bytes, os.PathLike]):
    """Download and extract tar gz file to location

    Arguments
        url - The URL to download
        path - The location to save results
    """
    with requests.get(url, stream=True) as rstream:
        rstream.raise_for_status()
        with tarfile.open(fileobj=rstream.raw, mode='r|gz') as tstream:
            tstream.extractall(path)


def zip(url: 'str', path: Union[str, bytes, os.PathLike]):
    """Download and extract zip file to location

    Arguments
        url - The URL to download
        path - The location to save results
    """
    with tempfile.NamedTemporaryFile() as tmp:
        file(url, tmp)
        with zipfile.ZipFile(tmp, 'r') as zfile:
            zfile.extractall(path)
