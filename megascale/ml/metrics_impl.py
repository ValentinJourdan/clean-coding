from typing import Any

import numpy as np
from scipy.stats import spearmanr

from megascale.ml.metrics import MetricsEvaluator


class SpearmanCorrEvaluator(MetricsEvaluator):
    """Implementation of the metrics evaluator for the Spearman correlation."""

    def name(self) -> str:
        return "Spearman correlation"

    def metric_computation(self, truth: np.ndarray, pred: np.ndarray) -> Any:
        return spearmanr(pred, truth)[0]
