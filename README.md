<h1 align="center">torchutil</h1>
<div align="center">

[![PyPI](https://img.shields.io/pypi/v/torchutil.svg)](https://pypi.python.org/pypi/torchutil)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Downloads](https://static.pepy.tech/badge/torchutil)](https://pepy.tech/project/torchutil)

General utilities for developing deep learning projects using PyTorch

`pip install torchutil`
</div>


## Table of contents

- [Checkpoint](#checkpoint)
    * [`torchutil.checkpoint.best_path`](#torchutilcheckpointbest_path)
    * [`torchutil.checkpoint.latest_path`](#torchutilcheckpointlatest_path)
    * [`torchutil.checkpoint.load`](#torchutilcheckpointload)
    * [`torchutil.checkpoint.save`](#torchutilcheckpointsave)
- [Cuda](#cuda)
    * [`torchutil.cuda.utilization`](#torchutilcudautilization)
- [Download](#download)
    * [`torchutil.download.file`](#torchutildownloadfile)
    * [`torchutil.download.tarbz2`](#torchutildownloadtarbz2)
    * [`torchutil.download.targz`](#torchutildownloadtargz)
    * [`torchutil.download.zip`](#torchutildownloadzip)
- [Gradients](#gradients)
    * [`torchutil.gradients.stats`](#torchutilgradientsstats)
- [Inference](#inference)
    * [`torchutil.inference.context`](#torchutilinferencecontext)
- [Iterator](#iterator)
    * [`torchutil.iterator`](#torchutiliterator)
    * [`torchutil.multiprocess_iterator`](#torchutilmultiprocess_iterator)
- [Metrics](#metrics)
    * [`torchutil.metrics.Accuracy`](#torchutilmetricsaccuracy)
    * [`torchutil.metrics.Average`](#torchutilmetricsaverage)
    * [`torchutil.metrics.F1`](#torchutilmetricsf1)
    * [`torchutil.metrics.L1`](#torchutilmetricsl1)
    * [`torchutil.metrics.MeanStd`](#torchutilmetricsmeanstd)
    * [`torchutil.metrics.Precision`](#torchutilmetricsprecision)
    * [`torchutil.metrics.PearsonCorrelation`](#torchutilmetricspearsoncorrelation)
    * [`torchutil.metrics.Recall`](#torchutilmetricsrecall)
    * [`torchutil.metrics.RMSE`](#torchutilmetricsrmse)
- [Notify](#notify)
    * [`torchutil.notify`](#torchutilnotify)
- [Paths](#paths)
    * [`torchutil.paths.chdir`](#torchutilpathschdir)
    * [`torchutil.paths.measure`](#torchutilpathsmeasure)
    * [`torchutil.paths.purge`](#torchutilpathspurge)
- [Tensorboard](#tensorboard)
    * [`torchutil.tensorboard.update`](#torchutiltensorboardupdate)
- [Time](#time)
    * [`torchutil.time.context`](#torchutiltimecontext)
    * [`torchutil.time.results`](#torchutiltimeresults)
    * [`torchutil.time.reset`](#torchutiltimereset)


## Checkpoint

```python
import torch
import torchutil

# Checkpoint location
file = 'model.pt'

# Initialize PyTorch model
model = torch.nn.Sequential(torch.nn.Conv1d())

# Initialize optimizer
optimizer = torch.nn.Adam(model.parameters())

# Save
torchutil.checkpoint.save(file, model, optimizer, step=0, epoch=0)

# Load for training
model, optimizer, state = torchutil.checkpoint.load(file, model, optimizer)
step, epoch = state['step'], state['epoch']

# Load for inference
model, *_ = torchutil.checkpoint.load(file, model, optimizer)
```


### `torchutil.checkpoint.best_path`

```python
def best_path(
    directory: Union[str, bytes, os.PathLike],
    glob: str = '*.pt',
    best_fn: Callable = highest_score
) -> Tuple[Union[str, bytes, os.PathLike], float]:
    """Retrieve the path to the best checkpoint

    Arguments
        directory - The directory to search for checkpoint files
        glob - The regular expression matching checkpoints
        best_fn - Takes a list of checkpoint paths and returns the latest
                  Default assumes checkpoint names are training step count.

    Returns
        best_file - The filename of the checkpoint with the best score
        best_score - The corresponding score
    """
```


### `torchutil.checkpoint.latest_path`

```python
def latest_path(
        directory: Union[str, bytes, os.PathLike],
        glob: str = '*.pt',
        latest_fn: Callable = largest_number_filename,
    ) -> Union[str, bytes, os.PathLike]:
    """Retrieve the path to the most recent checkpoint in a directory

    Arguments
        directory - The directory to search for checkpoint files
        glob - The regular expression matching checkpoints
        latest_fn - Takes a list of checkpoint paths and returns the latest.
                    Default assumes checkpoint names are training step count.

    Returns
        The latest checkpoint in directory according to latest_fn
    """
```


### `torchutil.checkpoint.load`

```python
def load(
    file: Union[str, bytes, os.PathLike],
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    map_location: str = 'cpu') -> Tuple[
        torch.nn.Module,
        Union[None, torch.optim.Optimizer],
        Dict
    ]:
    """Load model checkpoint

    Arguments
        file - The checkpoint file
        model - The PyTorch model
        optimizer - Optional PyTorch optimizer for training
        map_location - The device to load the checkpoint on

    Returns
        model - The model with restored weights
        optimizer - Optional optimizer with restored parameters
        state - Additional values that the user defined during save
    """
```

### `torchutil.checkpoint.save`

```python
def save(
    file: Union[str, bytes, os.PathLike],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    accelerator: Optional[accelerate.Accelerator] = None,
    **kwargs):
    """Save training checkpoint to disk

    Arguments
        file - The checkpoint file
        model - The PyTorch model
        optimizer - The PyTorch optimizer
        accelerator - HuggingFace Accelerator for device management
        kwargs - Additional values to save
    """
```

## Cuda

```python
import torch
import torchutil

# Directory to write Tensorboard files
directory = 'tensorboard'

# Training step
step = 0

# Log VRAM utilization in MB to Tensorboard
torchutil.tensorboard.update(
    directory,
    step,
    scalars=torchutil.cuda.utilization(torch.device('cuda:0'), 'MB'))
```


### `torchutil.cuda.utilization`

```python
def utilization(device: torch.Device, unit: str ='B') -> Dict[str, float]:
    """Get the current VRAM utilization of a specified device

    Arguments
        device
            The device to query for VRAM utilization
        unit
            Unit of memory utilization (bytes to terabytes); default bytes

    Returns
        Allocated and reserved VRAM utilization in the specified unit
    """
```


## Download

### `torchutil.download.file`

```python
def file(url: 'str', path: Union[str, bytes, os.PathLike]):
    """Download file from url

    Arguments
        url - The URL to download
        path - The location to save results
    """
```


### `torchutil.download.tarbz2`

```python
def tarbz2(url: 'str', path: Union[str, bytes, os.PathLike]):
    """Download and extract tar bz2 file to location

    Arguments
        url - The URL to download
        path - The location to save results
    """
```


### `torchutil.download.targz`

```python
def targz(url: 'str', path: Union[str, bytes, os.PathLike]):
    """Download and extract tar gz file to location

    Arguments
        url - The URL to download
        path - The location to save results
    """
```


### `torchutil.download.zip`

```python
def zip(url: 'str', path: Union[str, bytes, os.PathLike]):
    """Download and extract zip file to location

    Arguments
        url - The URL to download
        path - The location to save results
    """
```


## Gradients

```python
import torch
import torchutil

# Directory to write Tensorboard files
directory = 'tensorboard'

# Training step
step = 0

# Setup model and optimizer
# ...

# Compute forward pass and loss
# ...

# Zero gradients
optimizer.zero_grad()

# Compute gradients
loss.backward()

# Monitor gradients on tensorboard
torchutil.tensorboard.update(
    directory,
    step,
    scalars=torchutil.gradients.stats(model))

# Apply gradient update
optimizer.step()
```


### `torchutil.gradients.stats`

```python
def stats(model: torch.nn.Module) -> Dict[str, float]:
    """Get gradient statistics

    Arguments
        model
            The torch model

    Returns
        The L2 norm, maximum, and minimum gradients
    """
```


## Inference

### `torchutil.inference.context`

```python
@contextlib.contextmanager
def context(model: torch.nn.Module, autocast: bool = True) -> None:
    """Inference-time handling of model training flag and optimizations

    Arguments
        model
            The torch model performing inference
        autocast
            Whether to use mixed precision
    """
```


## Iterator

```python
import time
import torchutil

def wait(seconds):
    time.sleep(seconds)

n = 8
iterable = range(n)

# Monitor single-process job
for i in torchutil.iterator(iterable, message='single-process'):
    wait(i)

# Monitor multi-process job
torchutil.multiprocess_iterator(wait, iterable, message='multi-process')
```


### `torchutil.iterator`

```python
def iterator(
    iterable: Iterable,
    message: Optional[str] = None,
    initial: int = 0,
    total: Optional[int] = None
) -> Iterable:
    """Create a tqdm iterator

    Arguments
        iterable
            Items to iterate over
        message
            Static message to display
        initial
            Position to display corresponding to index zero of iterable
        total
            Length of the iterable; defaults to len(iterable)

    Returns
        Monitored iterable
    """
```


### `torchutil.multiprocess_iterator`

```python
def multiprocess_iterator(
    process: Callable,
    iterable: Iterable,
    message: Optional[str] = None,
    initial: int = 0,
    total: Optional[int] = None,
    num_workers: int = os.cpu_count(),
    worker_chunk_size: Optional[int] = None
) -> List:
    """Create a multiprocess tqdm iterator

    Arguments
        process
            The single-argument function called by each multiprocess worker
        iterable
            Items to iterate over
        message
            Static message to display
        initial
            Position to display corresponding to index zero of iterable
        total
            Length of the iterable; defaults to len(iterable)
        num_workers
            Multiprocessing pool size; defaults to number of logical CPU cores
        worker_chunk_size
            Number of items sent to each multiprocessing worker

    Returns
        Return values of calling process on each item, in original order
    """
```


## Metrics

```python
import torch
import torchutil

# Define a custom, batch-updating loss metric
class Loss(torchutil.metrics.Average):
    def update(self, predicted, target):

        # Compute your loss and the number of elements to average over
        loss = ...
        count = ...

        super().update(loss, count)

# Instantiate metrics
loss = Loss()
rmse = torchutil.metrics.RMSE()

# Generator that produces batches of predicted and target tensors
iterable = ...

# Update metrics
for predicted_tensor, target_tensor in iterable:
    loss.update(predicted_tensor, target_tensor)
    rmse.update(predicted_tensor, target_tensor)

# Get results
print('loss': loss())
print('rmse': rmse())
```


### `torchutil.metrics.Accuracy`

```python
class Accuracy(Metric):
    """Batch-updating accuracy metric"""

    def __call__(self)-> float:
        """Retrieve the current accuracy value

        Returns
            The current accuracy value
        """

    def update(self, predicted: torch.Tensor, target: torch.Tensor) -> None:
        """Update accuracy

        Arguments
            predicted
                The model prediction
            target
                The corresponding ground truth
        """

    def reset(self) -> None:
        """Reset accuracy"""
```


### `torchutil.metrics.Average`

```python
class Average(Metric):
    """Batch-updating average metric"""

    def __call__(self)-> float:
        """Retrieve the current average value

        Returns
            The current average value
        """

    def update(self, values: torch.Tensor, count: int) -> None:
        """Update running average

        Arguments
            values
                The values to average
            target
                The number of values
        """

    def reset(self) -> None:
        """Reset running average"""
```


### `torchutil.metrics.F1`

```python
class F1(Metric):
    """Batch-updating F1 score"""

    def __call__(self) -> float:
        """Retrieve the current F1 value

        Returns
            The current F1 value
        """

    def update(self, predicted: torch.Tensor, target: torch.Tensor) -> None:
        """Update F1

        Arguments
            predicted
                The model prediction
            target
                The corresponding ground truth
        """

    def reset(self) -> None:
        """Reset F1"""
```


### `torchutil.metrics.L1`

```python
class L1(Metric):
    """Batch updating L1 score"""

    def __call__(self) -> float:
        """Retrieve the current L1 value

        Returns
            The current L1 value
        """

    def update(self, predicted, target) -> None:
        """Update L1

        Arguments
            predicted
                The model prediction
            target
                The corresponding ground truth
        """

    def reset(self) -> None:
        """Reset L1"""
```


### `torchutil.metrics.MeanStd`

```python
class MeanStd(Metric):
    """Batch updating mean and standard deviation"""

    def __call__(self) -> Tuple[float, float]:
        """Retrieve the current mean and standard deviation

        Returns
            The current mean and standard deviation
        """

    def update(self, values: torch.Tensor) -> None:
        """Update mean and standard deviation

        Arguments
            predicted
                The model prediction
            target
                The corresponding ground truth
        """

    def reset(self) -> None:
        """Reset mean and standard deviation"""
```


### `torchutil.metrics.PearsonCorrelation`

```python
class PearsonCorrelation(Metric):
    """Batch-updating Pearson correlation"""

    def __init__(
        self,
        predicted_mean: float,
        predicted_std: float,
        target_mean: float,
        target_std: float
    ) -> None:
        """
        Arguments
            predicted_mean - Mean of predicted values
            predicted_std - Standard deviation of predicted values
            target_mean - Mean of target values
            target_std - Standard deviation of target values
        """

    def __call__(self) -> float:
        """Retrieve the current correlation value

        Returns
            The current correlation value
        """

    def update(self, predicted, target) -> None:
        """Update Pearson correlation

        Arguments
            predicted
                The model prediction
            target
                The corresponding ground truth
        """

    def reset(self) -> None:
        """Reset Pearson correlation"""
```


### `torchutil.metrics.Precision`

```python
class Precision(Metric):
    """Batch-updating precision metric"""

    def __call__(self) -> float:
        """Retrieve the current precision value

        Returns
            The current precision value
        """

    def update(self, predicted: torch.Tensor, target: torch.Tensor) -> None:
        """Update precision

        Arguments
            predicted
                The model prediction
            target
                The corresponding ground truth
        """

    def reset(self) -> None:
        """Reset precision"""
```


### `torchutil.metrics.Recall`

```python
class Recall(Metric):
    """Batch-updating recall metric"""

    def __call__(self) -> float:
        """Retrieve the current recall value


            The current recall value
        """

    def update(self, predicted: torch.Tensor, target: torch.Tensor) -> None:
        """Update recall

        Arguments
            predicted
                The model prediction
            target
                The corresponding ground truth
        """

    def reset(self) -> None:
        """Reset recall"""
```


### `torchutil.metrics.RMSE`

```python
class RMSE(Metric):
    """Batch-updating RMSE metric"""

    def __call__(self) -> float:
        """Retrieve the current rmse value

        Returns
            The current rmse value
        """

    def update(self, predicted: torch.Tensor, target: torch.Tensor) -> None:
        """Update RMSE

        Arguments
            predicted
                The model prediction
            target
                The corresponding ground truth
        """

    def reset(self) -> None:
        """Reset RMSE"""
```


## Notify

To use the `torchutil` notification system, set the `PYTORCH_NOTIFICATION_URL`
environment variable to a supported webhook as explained in
[the Apprise documentation](https://pypi.org/project/apprise/).

```python
import torchutil

# Send notification when function returns
@torchutil.notify('train')
def train():
    ...

# Equivalent using context manager
def train():
    with torchutil.notify('train'):
        ...
```


### `torchutil.notify`

```python
@contextlib.contextmanager
def notify(
    description: str,
    track_time: bool = True,
    notify_on_fail: bool = True):
    """Context manager for sending job notifications

    Arguments
        description - The name of the job being run
        track_time - Whether to report time elapsed
        notify_on_fail - Whether to send a notification on failure
    """
```


## Paths

### `torchutil.paths.chdir`

```python
@contextlib.contextmanager
def chdir(directory: Union[str, bytes, os.PathLike]) -> None:
    """Context manager for changing the current working directory

    Arguments
        directory
            The desired working directory
    """
```

This function is both a context manager and decorator.

```python
import tempfile
from pathlib import Path

import torchutil

# Create a directory
directory = tempfile.TemporaryDirectory()

# Create a file
file = 'tmp.txt'
(Path(directory.name) / file).touch()

# File is not in current working directory
assert not Path(file).exists()

# Change working directory using context manager
with torchutil.paths.chdir(directory.name):
    assert Path(file).exists()

# File is not in current working directory
assert not Path(file).exists()

# Change working directory using decorator
@torchutil.paths.chdir(directory.name)
def exists(file):
    assert Path(file).exists()
exists(file)

# File is not in current working directory
assert not Path(file).exists()

# Remove temporary paths
directory.cleanup()
```


### `torchutil.paths.measure`

```python
def measure(
    globs: Optional[List[Union[str, List[str]]]] = None,
    roots: Optional[
        List[
            Union[
                Union[str, bytes, os.PathLike],
                List[Union[str, bytes, os.PathLike]]
            ]
        ]
    ] = None,
    recursive: bool = False,
    unit='B'
) -> Union[int, float]:
    """Measure data usage of files and directories

    Arguments
        globs
            Globs matching paths to measure
        roots
            Directories to apply glob searches; current directory by default
        recursive
            Apply globs to all subdirectories of root directories
        unit
            Unit of memory utilization (bytes to terabytes); default bytes

    Returns
        Data usage in the specified unit
    """
```

This function also has a command-line interface.

```
python -m torchutil.paths.measure \
    [-h] \
    --globs GLOBS \
    [--roots ROOTS] \
    [--recursive] \
    [--unit]

Measure data usage of files and directories

arguments:
  --globs GLOBS
    Globs matching paths to measure

optional arguments:
  -h, --help
    show this help message and exit
  --roots ROOTS
    Directories to apply glob searches; current directory by default
  --recursive
    Apply globs to all subdirectories of root directories
  --unit
    Unit of memory utilization (bytes to terabytes); default bytes
```


### `torchutil.paths.purge`

```python
def purge(
    globs: Optional[List[Union[str, List[str]]]] = None,
    roots: Optional[
        List[
            Union[
                Union[str, bytes, os.PathLike],
                List[Union[str, bytes, os.PathLike]]
            ]
        ]
    ] = None,
    recursive: bool = False,
    force: bool = False
) -> None:
    """Remove all files and directories within directory matching glob

    Arguments
        globs
            Globs matching paths to delete
        roots
            Directories to apply glob searches; current directory by default
        recursive
            Apply globs to all subdirectories of root directories
        force
            Skip user confirmation of deletion
    """
```

This function also has a command-line interface.

```
python -m torchutil.paths.purge \
    [-h] \
    --globs GLOBS \
    [--roots ROOTS] \
    [--recursive] \
    [--force]

Remove files and directories

arguments:
  --globs GLOBS
    Globs matching paths to delete

optional arguments:
  -h, --help
    show this help message and exit
  --roots ROOTS
    Directories to apply glob searches; current directory by default
  --recursive
    Apply globs to all subdirectories of root directories
  --force
    Skip user confirmation of deletion
```


## Tensorboard

```python
import matplotlib.pyplot as plt
import torch
import torchutil

# Directory to write Tensorboard files
directory = 'tensorboard'

# Training step
step = 0

# Example audio
audio = torch.zeros(1, 16000)
sample_rate = 16000

# Example figure
figure = plt.figure()
plt.plot([0, 1, 2, 3])

# Example image
image = torch.zeros(256, 256, 3)

# Example scalar
loss = 0

# Update Tensorboard
torchutil.tensorboard.update(
    directory,
    step,
    audio={'audio': audio},
    sample_rate=sample_rate,
    figures={'figure': figure},
    images={'image': image},
    scalars={'loss': loss})
```


### `torchutil.tensorboard.update`

```python
def update(
    directory: Union[str, bytes, os.PathLike],
    step: int,
    audio: Optional[Dict[str, torch.Tensor]] = None,
    sample_rate: Optional[int] = None,
    figures: Optional[Dict[str, matplotlib.figure.Figure]] = None,
    images: Optional[Dict[str, torch.Tensor]] = None,
    scalars: Optional[Dict[str, Union[float, int, torch.Tensor]]] = None):
    """Update Tensorboard

    Arguments
        directory - Directory to write Tensorboard files
        step - Training step
        audio - Optional dictionary of 2D audio tensors to monitor
        sample_rate - Audio sample rate; required if audio is not None
        figures - Optional dictionary of Matplotlib figures to monitor
        images - Optional dictionary of 3D image tensors to monitor
        scalars - Optional dictionary of scalars to monitor
    """
```


## Time

```python
import time
import torchutil

# Perform timing
with torchutil.time.context('outer'):
    time.sleep(1)
    for i in range(2):
        time.sleep(1)
        with torchutil.time.context('inner'):
            time.sleep(1)

# Prints {'inner': 2.0020763874053955, 'outer': 5.005248308181763, 'total': 5.005248308181763}
print(torchutil.time.results())
```


### `torchutil.time.context`

```python
@contextlib.contextmanager
def context(name: str):
    """Wrapper to handle context changes of global timer

    Arguments
        name - Name of the timer to add time to
    """
```


### `torchutil.time.reset`

```python
def reset():
    """Clear timer state"""
```


### `torchutil.time.results`

```python
def results() -> dict:
    """Get timing results

    Returns
        Timing results: {name: elapsed_time} for all names
    """
```
