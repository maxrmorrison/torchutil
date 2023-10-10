import requests
import tarfile
import zipfile


###############################################################################
# Data download utilities
###############################################################################


def file(url, path):
    """Download file from url"""
    with requests.get(url, stream=True) as rstream:
        rstream.raise_for_status()
        with open(path, 'wb') as fstream:
            for chunk in rstream.iter_content(chunk_size=128):
                fstream.write(chunk)


def tarbz2(url, path):
    """Download and extract tar file to location"""
    with requests.get(url, stream=True) as rstream:
        rstream.raise_for_status()
        with tarfile.open(fileobj=rstream.raw, mode='r|bz2') as tstream:
            tstream.extractall(path)


def targz(url, path):
    """Download and extract tar file to location"""
    with requests.get(url, stream=True) as rstream:
        rstream.raise_for_status()
        with tarfile.open(fileobj=rstream.raw, mode='r|gz') as tstream:
            tstream.extractall(path)


def zip(url, path):
    """Download and extract zip file to location"""
    with requests.get(url, stream=True) as rstream:
        rstream.raise_for_status()
        with zipfile.ZipFile(rstream) as zstream:
            zstream.extractall(path)
