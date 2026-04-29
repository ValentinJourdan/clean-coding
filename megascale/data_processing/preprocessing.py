from dataclasses import dataclass

import pandas as pd
import numpy as np

from megascale.data_processing.environment import (
    construct_list_of_sequence_environments,
)
from megascale.data_processing.encoding import construct_feature_matrix, ENCODINGS


@dataclass
class PreprocessedData:
    train_features: np.ndarray
    train_scores: np.ndarray
    validation_features: np.ndarray
    validation_scores: np.ndarray
    test_features: np.ndarray
    test_scores: np.ndarray


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
) -> PreprocessedData:
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
    return PreprocessedData(
        train_features=_preprocess_split(train_data, emb_t),
        train_scores=np.array(train_data["score"]),
        validation_features=_preprocess_split(validation_data, emb_t),
        validation_scores=np.array(validation_data["score"]),
        test_features=_preprocess_split(test_data, emb_t),
        test_scores=np.array(test_data["score"]),
    )
