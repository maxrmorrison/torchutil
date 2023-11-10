from itertools import repeat
from pathlib import Path
from typing import List, Union, Optional
import os

def gather(
    *sources: List[Union[str, bytes, os.PathLike]],
    sinks: Optional[List[Union[str, bytes, os.PathLike]]] = None,
    source_extensions: Optional[str] = None,
    sink_extension: str = '.pt'):
    """
    Gathers lists of input and output directories and files into two lists
    of files, using the provided extension to glob directories.
    """
    if len(sources) == 0:
        raise ValueError('At least one source list must be provided')
    elif len(sources) == 1:
        assert sources[0] is not None, 'At least one source list must be provieded, but found None'
    else:
        assert source_extensions is None

    sources = list(sources)
    for i in range(0, len(sources)):
        if sources[i] is None:
            sources[i] = repeat(None)

    # Standardize extensions
    if source_extensions is not None:
        source_extensions = set([
            '.' + ext if '.' not in ext else ext
            for ext in source_extensions])
    else:
        source_extensions = []
    sink_extension = (
        '.' + sink_extension if '.' not in sink_extension
        else sink_extension)

    lengths = set()
    for source_list in sources:
        lengths.add(len(source_list))

    assert len(lengths) == 1, 'all source lists must have the same lengths'

    if sinks is not None:
        assert len(sinks) == len(sources[0]), 'sinks must have the same length as the source lists'

    if sinks is None:

        # Get sources as a list of files
        source_files = [[] for _ in sources]
        for source_tuple in zip(*sources):
            source_paths = [Path(source) for source in source_tuple]
            source_is_dir = list(set([source.is_dir() for source in source_paths]))
            if len(source_is_dir) > 1 and True in source_is_dir:
                raise ValueError('cannot handle directories when more than one input list is given')
            elif len(source_is_dir) == 1 and source_is_dir[0]:
                for extension in source_extensions:
                    source_files[0] += list(source_paths[0].rglob(f'*{extension}'))
            else:
                for i, source in enumerate(source_paths):
                    source_files[i].append(source)

        # Sink files are source files with sink extension
        sink_files = [
            file.with_suffix(sink_extension) for file in source_files]

    else:

        # Get sources and sinks as file lists
        source_files, sink_files = [[] for _ in sources], []
        for source_tuple, sink in zip(zip(*sources), sinks):
            source_paths = [Path(source) for source in source_tuple]
            source_is_dir = list(set([source.is_dir() for source in source_paths]))
            sink = Path(sink)

            if len(source_is_dir) > 1 and True in source_is_dir:
                raise ValueError('cannot handle directories when more than one input list is given')

            # Handle input directory (only if one source list)
            if True in source_is_dir:
                if not sink.is_dir():
                    raise RuntimeError(
                        f'For input tuple {source_tuple}, corresponding '
                        f'output {sink} is not a directory')
                for extension in source_extensions:
                    source_files[0] += list(source_tuple[0].rglob(f'*{extension}'))

                # Ensure one-to-one
                source_stems = [file.stem for file in source_files[0]]
                if not len(source_stems) == len(set(source_stems)):
                    raise ValueError(
                        'Two or more files have the same '
                        'stem with different extensions')

                # Get corresponding output files
                sink_files += [
                    sink / (file.stem + sink_extension)
                    for file in source_files]

            # Handle input file(s)
            else:
                if sink.is_dir():
                    raise RuntimeError(
                        f'For input file {source}, corresponding '
                        f'output {sink} is a directory')
                for i, source in enumerate(source_tuple):
                    source_files[i].append(source)
                sink_files.append(sink)

    if len(source_files) == 1:
        source_files = source_files[0]

    return source_files, sink_files
