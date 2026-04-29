import logging
from dataclasses import dataclass
from typing import Any, Callable, Union

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.core.scope import FrozenVariableDict
from insta_fs import FilesystemManager
from insta_fs.io import save_pkl

from megascale.ml.metrics_impl import SpearmanCorrEvaluator
from megascale.ml.model import OneDimensionalCNNModel

RANDOM_SEED = 42


@dataclass
class TrainingData:
    train_features: np.ndarray
    train_scores: np.ndarray
    validation_features: np.ndarray
    validation_scores: np.ndarray


@dataclass
class ModelConfig:
    conv_features: int
    dense_features: int
    use_pooling: bool
    conv_kernel_size: int = 3
    pooling_window_size: int = 3
    dense_bottleneck_factor: int = 10
    output_size: int = 1


@dataclass
class TrainingConfig:
    n_epochs: int
    batch_size: int
    optimizer_type: str
    optimizer_learning_rate: float


class Training:
    """Classes that manages the training of the 1D-CNN model."""

    def __init__(self, model_config: ModelConfig, init_data: np.ndarray):
        """Constructor

        Args:
            model_config: Architecture hyperparameters for the CNN model.
            init_data: Some data to initialize the parameters of the model with.
        """
        self._cnn_model = OneDimensionalCNNModel(
            conv_features=model_config.conv_features,
            dense_features=model_config.dense_features,
            use_pooling=model_config.use_pooling,
            conv_kernel_size=model_config.conv_kernel_size,
            pooling_window_size=model_config.pooling_window_size,
            dense_bottleneck_factor=model_config.dense_bottleneck_factor,
            output_size=model_config.output_size,
        )
        self._params: FrozenVariableDict | dict[str, Any] = self._cnn_model.init(
            jax.random.PRNGKey(RANDOM_SEED), init_data
        )

    def get_model_and_params(
        self,
    ) -> tuple[OneDimensionalCNNModel, Union[FrozenVariableDict, dict[str, Any]]]:
        """Returns current model and parameters of this model."""
        return self._cnn_model, self._params

    def _create_optimizer(
        self, optimizer_type: str, learning_rate: float
    ) -> optax.GradientTransformation:
        if optimizer_type == "adam":
            return optax.adam(learning_rate)
        elif optimizer_type == "sgd":
            return optax.sgd(learning_rate)
        else:
            raise ValueError(f"Unknown optimizer type: '{optimizer_type}'")

    def _train_one_epoch(
        self,
        train_features: np.ndarray,
        train_scores: np.ndarray,
        batch_size: int,
        update_fn: Callable,
        opt_state: Any,
    ) -> tuple[list, Any]:
        losses = []
        for i in range(0, train_features.shape[0], batch_size):
            self._params, opt_state, loss = update_fn(
                self._params,
                opt_state,
                train_features[i : i + batch_size],
                train_scores[i : i + batch_size],
            )
            losses.append(loss)
        return losses, opt_state

    def _evaluate(
        self,
        validation_features: np.ndarray,
        validation_scores: np.ndarray,
        spearman_eval: SpearmanCorrEvaluator,
    ) -> float:
        validation_pred = np.asarray(
            self._cnn_model.apply(self._params, validation_features)
        )
        return spearman_eval.metric_computation(validation_pred, validation_scores)

    def _save_model(self, s3_loc: str) -> None:
        FilesystemManager().s3_init()
        save_pkl(self._params, s3_loc)

    def run(
        self,
        data: TrainingData,
        config: TrainingConfig,
        spearman_eval: SpearmanCorrEvaluator,
        s3_loc: str,
    ) -> None:
        """Runs a training loop.

        Args:
            data: Training and validation features and scores.
            config: Hyperparameters for the training loop.
            spearman_eval: Evaluator for the Spearman correlation.
            s3_loc: URI string of where to store the final parameters to.

        """

        optimizer = self._create_optimizer(config.optimizer_type, config.optimizer_learning_rate)

        @jax.jit
        def _compute_loss(params, features, target):
            y_pred = self._cnn_model.apply(params, features)
            loss = jnp.mean(optax.l2_loss(jnp.asarray(y_pred).flatten(), target))
            return loss

        @jax.jit
        def _update_step(params, opt_state, features, target):
            loss, grads = jax.value_and_grad(_compute_loss)(params, features, target)
            updates, opt_state = optimizer.update(grads, opt_state)
            params = optax.apply_updates(params, updates)
            return params, opt_state, loss

        opt_state = optimizer.init(self._params)
        best_corr = 0
        best_params: FrozenVariableDict | dict[str, Any] = self._params
        best_epoch = 0

        for epoch_idx in range(config.n_epochs):
            losses, opt_state = self._train_one_epoch(
                data.train_features, data.train_scores, config.batch_size, _update_step, opt_state
            )
            spearman_corr = self._evaluate(
                data.validation_features, data.validation_scores, spearman_eval
            )

            if best_corr < spearman_corr:
                best_corr = spearman_corr
                best_params = self._params
                best_epoch = epoch_idx

            logging.info(
                "[epoch %i] Corr: %.3f - Loss: %.3f - Best epoch so far: %i",
                epoch_idx + 1,
                spearman_corr,
                np.mean(losses),
                best_epoch + 1,
            )

        self._params = best_params
        self._save_model(s3_loc)
