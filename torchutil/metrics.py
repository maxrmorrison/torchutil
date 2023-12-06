import math
from abc import ABC, abstractmethod
from typing import Tuple

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

        Returns
            The current accuracy value
        """
        return (self.true_positives / self.count).item()

    def update(self, predicted: torch.Tensor, target: torch.Tensor) -> None:
        """Update accuracy

        Arguments
            predicted
                The model prediction
            target
                The corresponding ground truth
        """
        self.count += predicted.numel()
        self.true_positives += (predicted == target).sum()

    def reset(self) -> None:
        """Reset accuracy"""
        self.count = 0
        self.true_positives = 0


class Average(Metric):
    """Batch-updating average metric"""

    def __init__(self) -> None:
        self.reset()

    def __call__(self)-> float:
        """Retrieve the current average value

        Returns
            The current average value
        """
        return (self.total / self.count).item()

    def update(self, values: torch.Tensor, count: int) -> None:
        """Update running average

        Arguments
            values
                The values to average
            count
                The number of values
        """
        self.count += count
        self.total += values.sum()

    def reset(self) -> None:
        """Reset running average"""
        self.count = 0
        self.total = 0.


class F1(Metric):
    """Batch-updating F1 score"""

    def __init__(self) -> None:
        self.precision = Precision()
        self.recall = Recall()

    def __call__(self) -> float:
        """Retrieve the current F1 value

        Returns
            The current F1 value
        """
        precision, recall = self.precision(), self.recall()
        try:
            return 2 * precision * recall / (precision + recall)
        except ZeroDivisionError:
            return 0

    def update(self, predicted: torch.Tensor, target: torch.Tensor) -> None:
        """Update F1

        Arguments
            predicted
                The model prediction
            target
                The corresponding ground truth
        """
        self.precision.update(predicted, target)
        self.recall.update(predicted, target)

    def reset(self) -> None:
        """Reset F1"""
        self.precision.reset()
        self.recall.reset()


class L1(Metric):
    """Batch updating L1 score"""

    def __init__(self) -> None:
        self.reset()

    def __call__(self) -> float:
        """Retrieve the current L1 value

        Returns
            The current L1 value
        """
        return (self.total / self.count).item()

    def update(self, predicted, target) -> None:
        """Update L1

        Arguments
            predicted
                The model prediction
            target
                The corresponding ground truth
        """
        self.count += predicted.numel()
        self.total += torch.abs(predicted - target).sum()

    def reset(self) -> None:
        """Reset L1"""
        self.count = 0
        self.total = 0.


class MeanStd(Metric):
    """Batch updating mean and standard deviation"""

    def __init__(self) -> None:
        self.reset()

    def __call__(self) -> Tuple[float, float]:
        """Retrieve the current mean and standard deviation

        Returns
            The current mean and standard deviation
        """
        std = math.sqrt(self.m2 / (self.count - 1))
        return self.mean, std

    def update(self, values: torch.Tensor) -> None:
        """Update mean and standard deviation

        Arguments
            predicted
                The model prediction
            target
                The corresponding ground truth
        """
        for value in values:
            self.count += 1
            delta = value - self.mean
            self.mean += delta / self.count
            delta2 = value - self.mean
            self.m2 += delta * delta2

    def reset(self) -> None:
        """Reset mean and standard deviation"""
        self.m2 = 0.
        self.mean = 0.
        self.count = 0


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
        self.reset()
        self.predicted_mean = predicted_mean
        self.predicted_std = predicted_std
        self.target_mean = target_mean
        self.target_std = target_std

    def __call__(self) -> float:
        """Retrieve the current correlation value

        Returns
            The current correlation value
        """
        return (
            1. / (self.predicted_std * self.target_std + 1e-6) *
            (self.total / self.count).item())

    def update(self, predicted, target) -> None:
        """Update Pearson correlation

        Arguments
            predicted
                The model prediction
            target
                The corresponding ground truth
        """
        self.total += sum(
            (predicted - self.predicted_mean) * (target - self.target_mean))
        self.count += predicted.numel()

    def reset(self) -> None:
        """Reset Pearson correlation"""
        self.count = 0
        self.total = 0.


class Precision(Metric):
    """Batch-updating precision metric"""

    def __init__(self) -> None:
        self.reset()

    def __call__(self) -> float:
        """Retrieve the current precision value

        Returns
            The current precision value
        """
        denominator = self.true_positives + self.false_positives
        return (self.true_positives / denominator).item()

    def update(self, predicted: torch.Tensor, target: torch.Tensor) -> None:
        """Update precision

        Arguments
            predicted
                The model prediction
            target
                The corresponding ground truth
        """
        self.true_positives += (predicted & target).sum()
        self.false_positives += (predicted & ~target).sum()

    def reset(self) -> None:
        """Reset precision"""
        self.true_positives = 0
        self.false_positives = 0


class Recall(Metric):
    """Batch-updating recall metric"""

    def __init__(self) -> None:
        self.reset()

    def __call__(self) -> float:
        """Retrieve the current recall value

        Returns
            The current recall value
        """
        denominator = self.true_positives + self.false_negatives
        return (self.true_positives / denominator).item()

    def update(self, predicted: torch.Tensor, target: torch.Tensor) -> None:
        """Update recall

        Arguments
            predicted
                The model prediction
            target
                The corresponding ground truth
        """
        self.true_positives += (predicted & target).sum()
        self.false_negatives += (~predicted & target).sum()

    def reset(self) -> None:
        """Reset recall"""
        self.true_positives = 0
        self.false_negatives = 0


class RMSE(Metric):
    """Batch-updating RMSE metric"""

    def __init__(self) -> None:
        self.reset()

    def __call__(self) -> float:
        """Retrieve the current rmse value

        Returns
            The current rmse value
        """
        return math.sqrt((self.total / self.count).item())

    def update(self, predicted: torch.Tensor, target: torch.Tensor) -> None:
        """Update RMSE

        Arguments
            predicted
                The model prediction
            target
                The corresponding ground truth
        """
        self.count += predicted.numel()
        self.total += ((predicted - target) ** 2).sum()

    def reset(self) -> None:
        """Reset RMSE"""
        self.count = 0
        self.total = 0.
