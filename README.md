<h1 align="center">torchutil</h1>
<div align="center">

[![PyPI](https://img.shields.io/pypi/v/torchutil.svg)](https://pypi.python.org/pypi/torchutil)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Downloads](https://pepy.tech/badge/torchutil)](https://pepy.tech/project/torchutil)

</div>

PyTorch utilities for developing deep learning frameworks


## Table of contents

- [Checkpoint](#checkpoint)
    * [`torchutil.checkpoint.latest_path`](torchutilcheckpointlatest_path)
    * [`torchutil.checkpoint.load`](torchutilcheckpointload)
    * [`torchutil.checkpoint.save`](torchutilcheckpointsave)
- [Download](#download)
    * [`torchutil.download.file`](torchutildownloadfile)
    * [`torchutil.download.tarbz2`](torchutildownloadtarbz2)
    * [`torchutil.download.targz`](torchutildownloadtargz)
    * [`torchutil.download.zip`](torchutildownloadzip)
- [Notify](#notify)
    * [`torchutil.notify.on_finish`](torchutilnotifyon_finish)
- [Tensorboard](#tensorboard)
    * [`torchutil.tensorboard.update`](torchutiltensorboardupdate)
- [Time](#time)
    * [`torchutil.time.context`](torchutiltimecontext)
    * [`torchutil.time.results`](torchutiltimeresults)


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


### `torchutil.checkpoint.latest_path`


### `torchutil.checkpoint.load`


### `torchutil.checkpoint.save`


## Download


### `torchutil.download.file`


### `torchutil.download.tarbz2`


### `torchutil.download.targz`


### `torchutil.download.zip`


## Notify

```python
import torchutil

# Send notification when training finishes
@torchutil.notify.on_finish('train')
def train():
    ...

# Equivalent using "with"
def train():
    with torchutil.notify.on_finish('train):
        ...
```


### `torchutil.notify.on_finish`

```python
@contextlib.contextmanager
def on_finish(
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

# Perform timing
with torchutil.time.context('outer'):
    time.sleep(1)
    for i in range(2):
        time.sleep(1)
        with torchutil.time.context('inner'):
            time.sleep(1)

# Prints {'outer': TODO, 'inner': TODO}
print(torchutil.timer.results())
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


### `torchutil.time.results`

```python
def results() -> dict:
    """Get timing results

    Returns
        Timing results: {name: elapsed_time} for all names
    """
```
