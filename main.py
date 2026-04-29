from os.path import dirname
import logging

import numpy as np
import pandas as pd
import hydra
from omegaconf import DictConfig

from megascale.data_processing.split import get_data_splitting
from megascale.ml.training import Training, OptimizerConfig
from megascale.ml.metrics_impl import SpearmanCorrEvaluator
from megascale.data_processing.preprocessing import preprocess_data

DATA_PATH = dirname(__file__) + "/data/data.csv"


@hydra.main(version_base=None, config_path="config", config_name="main")
def main(config: DictConfig) -> None:
    """Main function to run the training of the model and evaluate on test set at end.

    Args:
        config: Configuration object from hydra.
    """
    data = pd.read_csv(DATA_PATH)
    train_data, validation_data, test_data, train_pr = get_data_splitting(data, 42)
    logging.info("Proportion of training data: %.3f", train_pr)

    preprocessed = preprocess_data(
        train_data, validation_data, test_data, config.features.embedding_type
    )

    spear = SpearmanCorrEvaluator()
    training = Training(
        config.model.conv_features,
        config.model.dense_features,
        not config.model.use_pooling,
        preprocessed.train_features[:5],
    )
    training.run(
        preprocessed,
        config.num_epochs,
        config.batch_size,
        OptimizerConfig(
            type=config.optimizer.type,
            learning_rate=config.optimizer.learning_rate,
        ),
        spear,
        config.s3_final_model,
    )
    opt_model, opt_params = training.get_model_and_params()

    test_pred = np.asarray(opt_model.apply(opt_params, preprocessed.test_features))
    spearman_corr = spear.metric_computation(test_pred, preprocessed.test_scores)
    logging.info("Final %s on test set: %.3f", spear.name(), spearman_corr)


if __name__ == "__main__":
    main()
