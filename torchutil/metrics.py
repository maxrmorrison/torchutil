import math
from abc import ABC, abstractmethod

import torch


###############################################################################
# Base batch-updating metric
###############################################################################


class Metric(ABC):
    """Base batch-updating metric"""

    @abstractmethod
    def __call__(self) -> float:
        """Retrieve the current metric values

        Returns
            The current metric value
        """
        pass

    @abstractmethod
    def update(self, predicted: torch.Tensor, target: torch.Tensor) -> None:
        """Update the metric

        Arguments
            predicted
                The model prediction
            target
                The corresponding ground truth
        """
        pass

    @abstractmethod
    def reset(self) -> None:
        """Reset the metric"""
        pass


###############################################################################
# Derived batch-updating metrics
###############################################################################


class Accuracy(Metric):
    """Batch-updating accuracy metric"""

    def __init__(self) -> None:
        self.reset()

    def __call__(self)-> float:
        """Retrieve the current accuracy value

        Returns:
            The current accuracy value
        """
        return (self.true_positives / self.count).item()

    def update(self, predicted: torch.Tensor, target: torch.Tensor) -> None:
        """Update the metric

        Arguments
            predicted
                The model prediction
            target
                The corresponding ground truth
        """
        self.count += predicted.numel()
        self.true_positives += (predicted == target).sum()

    def reset(self) -> None:
        """Reset the metric"""
        self.count = 0
        self.true_positives = 0


class Average(Metric):
    """Batch-updating average metric"""

    def __init__(self) -> None:
        self.reset()

    def __call__(self)-> float:
        """Retrieve the current average value

        Returns:
            The current average value
        """
        return (self.total / self.count).item()

    def update(self, values: torch.Tensor, count: int) -> None:
        """Update the metric

        Arguments
            values
                The values to average
            count
                The number of values
        """
        self.count += count
        self.total += values.sum()

    def reset(self) -> None:
        """Reset the metric"""
        self.count = 0
        self.total = 0.


class F1(Metric):
    """Batch-updating F1 score"""

    def __init__(self) -> None:
        self.precision = Precision()
        self.recall = Recall()

    def __call__(self) -> float:
        """Retrieve the current F1 value

        Returns:
            The current F1 value
        """
        precision, recall = self.precision(), self.recall()
        try:
            return 2 * precision * recall / (precision + recall)
        except ZeroDivisionError:
            return 0

    def update(self, predicted: torch.Tensor, target: torch.Tensor) -> None:
        """Update the metric

        Arguments
            predicted
                The model prediction
            target
                The corresponding ground truth
        """
        self.precision.update(predicted, target)
        self.recall.update(predicted, target)

    def reset(self) -> None:
        """Reset the metric"""
        self.precision.reset()
        self.recall.reset()


class L1(Metric):
    """Batch updating L1 score"""

    def __init__(self) -> None:
        self.reset()

    def __call__(self) -> float:
        """Retrieve the current L1 value

        Returns:
            The current L1 value
        """
        return (self.total / self.count).item()

    def update(self, predicted, target):
        """Update the metric

        Arguments
            predicted
                The model prediction
            target
                The corresponding ground truth
        """
        self.count += predicted.numel()
        self.total += torch.abs(predicted - target).sum()

    def reset(self) -> None:
        """Reset the metric"""
        self.count = 0
        self.total = 0.


class Precision(Metric):
    """Batch-updating precision metric"""

    def __init__(self) -> None:
        self.reset()

    def __call__(self) -> float:
        """Retrieve the current precision value

        Returns:
            The current precision value
        """
        denominator = self.true_positives + self.false_positives
        return (self.true_positives / denominator).item()

    def update(self, predicted: torch.Tensor, target: torch.Tensor) -> None:
        """Update the metric

        Arguments
            predicted
                The model prediction
            target
                The corresponding ground truth
        """
        self.true_positives += (predicted & target).sum()
        self.false_positives += (predicted & ~target).sum()

    def reset(self) -> None:
        """Reset the metric"""
        self.true_positives = 0
        self.false_positives = 0


class Recall(Metric):
    """Batch-updating recall metric"""

    def __init__(self) -> None:
        self.reset()

    def __call__(self) -> float:
        """Retrieve the current recall value

        Returns:
            The current recall value
        """
        denominator = self.true_positives + self.false_negatives
        return (self.true_positives / denominator).item()

    def update(self, predicted: torch.Tensor, target: torch.Tensor) -> None:
        """Update the metric

        Arguments
            predicted
                The model prediction
            target
                The corresponding ground truth
        """
        self.true_positives += (predicted & target).sum()
        self.false_negatives += (~predicted & target).sum()

    def reset(self) -> None:
        """Reset the metric"""
        self.true_positives = 0
        self.false_negatives = 0


class RMSE(Metric):
    """Batch-updating RMSE metric"""

    def __init__(self) -> None:
        self.reset()

    def __call__(self) -> float:
        """Retrieve the current rmse value

        Returns:
            The current rmse value
        """
        return math.sqrt((self.total / self.count).item())

    def update(self, predicted: torch.Tensor, target: torch.Tensor) -> None:
        """Update the metric

        Arguments
            predicted
                The model prediction
            target
                The corresponding ground truth
        """
        self.count += predicted.numel()
        self.total += ((predicted - target) ** 2).sum()

    def reset(self) -> None:
        """Reset the metric"""
        self.count = 0
        self.total = 0.
