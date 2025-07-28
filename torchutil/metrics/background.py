import math
from abc import ABC, abstractmethod
from typing import Tuple
import time

from multiprocessing import Process, Value

import torch
from torchutil.metrics import Metric


class BackgroundMetric(Metric):
    """Base metric that runs in a background process"""

    def __init__(self) -> None:
        super().__init__()
        self.process = None

    def __enter__(self):
        self.process = Process(target=self._update)
        self.process.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.process.terminate()
        self.process.join()

    def _update(self):
        while True:
            self.update()
            time.sleep(1)


if __name__ == '__main__':
    class Counter(BackgroundMetric):
        def __init__(self):
            self.counter = Value('i', 0)
            super().__init__()
            self.reset()

        def update(self):
            self.counter.value = self.counter.value + 1

        def reset(self):
            self.counter.value = 0

        def __call__(self):
            return self.counter.value

    counter = Counter()

    with counter:
        time.sleep(5)

    print(counter())