from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from megascale.ml.training import OptimizerConfig, Training


@pytest.fixture
def training():
    return Training(
        conv_features=4,
        dense_features=10,
        without_pooling=True,
        init_data=np.zeros((1, 11, 40)),
    )


def test_create_optimizer_adam_uses_correct_learning_rate(training):
    config = OptimizerConfig(type="adam", learning_rate=0.01)

    with patch("megascale.ml.training.optax.adam") as mock_adam:
        training._create_optimizer(config)

    mock_adam.assert_called_once_with(0.01)


def test_create_optimizer_sgd_uses_correct_learning_rate(training):
    config = OptimizerConfig(type="sgd", learning_rate=0.05)

    with patch("megascale.ml.training.optax.sgd") as mock_sgd:
        training._create_optimizer(config)

    mock_sgd.assert_called_once_with(0.05)


def test_create_optimizer_raises_for_unknown_type(training):
    config = OptimizerConfig(type="unknown", learning_rate=0.01)

    with pytest.raises(ValueError, match="Unknown optimizer type"):
        training._create_optimizer(config)


def test_run_keeps_params_from_best_correlation_epoch(training):
    epoch_params = [{"epoch_id": i} for i in range(4)]
    correlations = [0.3, 0.7, 0.5, 0.2]
    epoch_counter = {"i": 0}

    def fake_train_one_epoch(*args, **kwargs):
        idx = epoch_counter["i"]
        training._params = epoch_params[idx]
        epoch_counter["i"] += 1
        return [], None

    with patch.object(training, "_train_one_epoch", side_effect=fake_train_one_epoch), \
         patch.object(training, "_evaluate", side_effect=correlations), \
         patch.object(training, "_save_model"):
        training.run(
            data=MagicMock(),
            n_epochs=4,
            batch_size=32,
            optimizer_config=OptimizerConfig(type="adam", learning_rate=0.01),
            spearman_eval=MagicMock(),
            s3_loc="dummy",
        )

    assert training._params == epoch_params[1]
