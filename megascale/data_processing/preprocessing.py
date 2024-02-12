import pandas as pd
import numpy as np

from megascale.data_processing.environment import (
    construct_list_of_sequence_environments,
)
from megascale.data_processing.one_hot_encoding import (
    construct_feature_matrix_with_one_hot_encoding,
)
from megascale.data_processing.z_scales_encoding import (
    construct_feature_matrix_with_z_scales_encoding,
)


def preprocess_data(
    train_data: pd.DataFrame,
    validation_data: pd.DataFrame,
    test_data: pd.DataFrame,
    emb_t: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Preprocesses the data.

    This means creating features and targets for train/validation/test datasets.

    Args:
        train_data: Training data.
        validation_data: Validation data.
        test_data: Test data.
        emb_t: Embedding type for residues, either "zscales" or "one-hot".

    Returns:
        Features and targets for training, validation and test sets.

    """
    # Store target values
    train_t = np.array(train_data["score"])
    validation_t = np.array(validation_data["score"])
    test_t = np.array(test_data["score"])

    # Preprocess training set
    train_seq_envs = construct_list_of_sequence_environments(
        train_data["aa_seq"], train_data["variant"], 5
    )
    if emb_t == "one-hot":
        train_f = construct_feature_matrix_with_one_hot_encoding(
            train_seq_envs, [var[0] for var in train_data["variant"]]
        )
    elif emb_t == "zscales":
        train_f = construct_feature_matrix_with_z_scales_encoding(
            train_seq_envs, [var[0] for var in train_data["variant"]]
        )

    # Preprocess validation set
    validation_seq_envs = construct_list_of_sequence_environments(
        validation_data["aa_seq"], validation_data["variant"], 5
    )
    if emb_t == "one-hot":
        validation_f = construct_feature_matrix_with_one_hot_encoding(
            validation_seq_envs, [var[0] for var in validation_data["variant"]]
        )
    elif emb_t == "zscales":
        validation_f = construct_feature_matrix_with_z_scales_encoding(
            validation_seq_envs, [var[0] for var in validation_data["variant"]]
        )

    # Preprocess test set
    test_seq_envs = construct_list_of_sequence_environments(
        test_data["aa_seq"], test_data["variant"], 5
    )
    if emb_t == "one-hot":
        test_f = construct_feature_matrix_with_one_hot_encoding(
            test_seq_envs, [var[0] for var in test_data["variant"]]
        )
    elif emb_t == "zscales":
        test_f = construct_feature_matrix_with_z_scales_encoding(
            test_seq_envs, [var[0] for var in test_data["variant"]]
        )

    return train_f, train_t, validation_f, validation_t, test_f, test_t
