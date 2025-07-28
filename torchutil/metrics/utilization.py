from multiprocessing import Value
import psutil

from torchutil.metrics import BackgroundMetric

class RamUsage(BackgroundMetric):

    def __init__(self):
        self.total = Value('l', 0)
        self.count = Value('l', 0)
        self.max = Value('l', 0)

    def reset(self):
        self.total.value = 0
        self.count.value = 0
        self.max.value = 0

    def update(self):
        memory_statistics = psutil.virtual_memory()
        # Apparently this is more accurate than .used() because of cache, etc.
        total_used = memory_statistics.total - memory_statistics.available
        if total_used > self.max.value:
            self.max.value = total_used
        self.total.value += total_used
        self.count.value += 1

    def __call__(self):
        return {
            'average (GB)': self.total.value / self.count.value / 1024 / 1024 / 1024,
            'maximum (GB)': self.max.value / 1024 / 1024 / 1024
        }

if __name__ == '__main__':
    import time
    import torch
    ram = RamUsage()
    with ram:
        time.sleep(5)
        t = torch.zeros(100000)
        time.sleep(3)
        t = torch.randn(5000000000)
    print(ram())