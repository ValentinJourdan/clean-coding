from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class MetricsEvaluator(ABC):
    """Base class for all metrics evaluators."""

    @abstractmethod
    def name(self) -> str:
        """Returns the name of the metric."""

    @abstractmethod
    def metric_computation(self, truth: np.ndarray, pred: np.ndarray) -> Any:
        """Calculates the metric and returns it.

        Args:
            truth: Ground truth values.
            pred: Predicted values.

        Returns:
            The metric value that was computed.
        """
