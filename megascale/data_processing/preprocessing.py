import pandas as pd
import numpy as np

from megascale.data_processing.environment import (
    construct_list_of_sequence_environments,
)
from megascale.data_processing.encoding import construct_feature_matrix, ENCODINGS


def _preprocess_split(data: pd.DataFrame, emb_t: str) -> np.ndarray:
    if emb_t not in ENCODINGS:
        raise ValueError(f"Unknown embedding type: '{emb_t}'")
    seq_envs = construct_list_of_sequence_environments(
        data["aa_seq"], data["variant"], 5
    )
    wild_types = [var[0] for var in data["variant"]]
    return construct_feature_matrix(seq_envs, wild_types, ENCODINGS[emb_t])


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
    train_t = np.array(train_data["score"])
    validation_t = np.array(validation_data["score"])
    test_t = np.array(test_data["score"])

    train_f = _preprocess_split(train_data, emb_t)
    validation_f = _preprocess_split(validation_data, emb_t)
    test_f = _preprocess_split(test_data, emb_t)

    return train_f, train_t, validation_f, validation_t, test_f, test_t
