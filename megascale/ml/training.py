import logging
from typing import Any, Union

from flax.core.scope import FrozenVariableDict
import jax
import jax.numpy as jnp
import numpy as np
import optax
from insta_fs import FilesystemManager  # ty: ignore[unresolved-import]
from insta_fs.io import save_pkl  # ty: ignore[unresolved-import]

from megascale.ml.model import OneDimensionalCNNModel
from megascale.ml.metrics_impl import SpearmanCorrEvaluator


class Training:
    """Classes that manages the training of the 1D-CNN model."""

    def __init__(
        self,
        conv_features: int,
        dense_features: int,
        without_pooling: bool,
        init_data: np.ndarray,
    ):
        """Constructor.

        Args:
            conv_features: Number of features for convolution layers model.
            dense_features: Numer of features for dense layers of model.
            without_pooling: Whether pooling should be applied after the conv. layers
                             of the model.
            init_data: Some data to initialize the parameters of the model with.
        """
        self._cnn_model = OneDimensionalCNNModel(
            conv_features=conv_features,
            dense_features=dense_features,
            without_pooling=without_pooling,
        )
        self._params: FrozenVariableDict | dict[str, Any] = self._cnn_model.init(
            jax.random.PRNGKey(42), init_data
        )

    def get_model_and_params(
        self,
    ) -> tuple[OneDimensionalCNNModel, Union[FrozenVariableDict, dict[str, Any]]]:
        """Returns current model and parameters of this model."""
        return self._cnn_model, self._params

    def run(
        self,
        train_features: np.ndarray,
        train_scores: np.ndarray,
        validation_features: np.ndarray,
        validation_scores: np.ndarray,
        n_epochs: int,
        batch_size: int,
        optimizer_type: str,
        optimizer_learning_rate: float,
        spearman_eval: SpearmanCorrEvaluator,
        s3_loc: str,
    ) -> None:
        """Runs a training loop.

        Args:
            train_features: Training features.
            train_scores: Training targets.
            validation_features: Validation features.
            validation_scores: Validation targets.
            n_epochs: Number of epochs.
            batch_size: Number of data points per batch.
            optimizer_type: The type of the optimizer to use.
            optimizer_learning_rate: The learning rate for the optimizer.
            spearman_eval: Evaluator for the Spearman correlation.
            s3_loc: URI string of where to store the final parameters to.

        """

        @jax.jit
        def _compute_loss(params, features, target):
            y_pred = self._cnn_model.apply(params, features)
            loss = jnp.mean(optax.l2_loss(jnp.asarray(y_pred).flatten(), target))
            return loss

        @jax.jit
        def _update_step(params, opt_state, features, target):
            loss, grads = jax.value_and_grad(_compute_loss)(
                params,
                features,
                target,
            )
            updates, opt_state = optimizer.update(grads, opt_state)
            params = optax.apply_updates(params, updates)
            return params, opt_state, loss

        if optimizer_type == "adam":
            optimizer = optax.adam(optimizer_learning_rate)
        elif optimizer_type == "sgd":
            optimizer = optax.sgd(optimizer_learning_rate)

        opt_state = optimizer.init(self._params)

        best_corr = 0
        best_params: FrozenVariableDict | dict[str, Any] = self._params
        best_epoch = 0

        for epoch_idx in range(n_epochs):
            losses = []
            for i in range(0, train_features.shape[0], batch_size):
                start_idx = i
                end_idx = i + batch_size
                self._params, opt_state, loss = _update_step(
                    self._params,
                    opt_state,
                    train_features[start_idx:end_idx],
                    train_scores[start_idx:end_idx],
                )
                losses.append(loss)

            validation_pred = np.asarray(
                self._cnn_model.apply(self._params, validation_features)
            )
            spearman_corr = spearman_eval.metric_computation(
                validation_pred, validation_scores
            )

            if best_corr < spearman_corr:
                best_corr = spearman_corr
                best_params = self._params
                best_epoch = epoch_idx

            avg_loss = 0
            for loss in losses:
                avg_loss += loss
            avg_loss /= len(losses)

            logging.info(
                "[epoch %i] Corr: %.3f - Loss: %.3f - Best epoch so far: %i",
                epoch_idx + 1,
                spearman_corr,
                avg_loss,
                best_epoch + 1,
            )

        self._params = best_params
        FilesystemManager().s3_init()
        save_pkl(self._params, s3_loc)
