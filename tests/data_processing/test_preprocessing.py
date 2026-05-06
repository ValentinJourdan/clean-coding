from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from megascale.data_processing.preprocessing import _preprocess_split, preprocess_data


def _make_dummy_df(pdbid: str, n: int = 3) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "aa_seq": ["ACDEFGHIKL"] * n,
            "variant": ["A1G"] * n,
            "score": [0.5] * n,
            "pdbid": [pdbid] * n,
        }
    )


def test_preprocess_data_raises_for_unknown_embedding_type():
    df = _make_dummy_df("pdb1")
    with pytest.raises(ValueError, match="Unknown embedding type"):
        preprocess_data(df, df, df, "unknown")


def test_preprocess_data_returns_correct_number_of_samples():
    train = _make_dummy_df("pdb1", n=4)
    valid = _make_dummy_df("pdb2", n=2)
    test = _make_dummy_df("pdb3", n=3)

    result = preprocess_data(train, valid, test, "one-hot")

    assert result.train_features.shape[0] == 4
    assert result.validation_features.shape[0] == 2
    assert result.test_features.shape[0] == 3


def test_preprocess_data_calls_preprocess_split_for_each_split():
    train = _make_dummy_df("pdb1", n=4)
    valid = _make_dummy_df("pdb2", n=2)
    test = _make_dummy_df("pdb3", n=3)
    dummy_features = np.zeros((1, 11, 40))

    with patch(
        "megascale.data_processing.preprocessing._preprocess_split",
        return_value=dummy_features,
    ) as mock_split:
        result = preprocess_data(train, valid, test, "one-hot")

    assert mock_split.call_count == 3
    passed_dataframes = [call.args[0] for call in mock_split.call_args_list]
    assert passed_dataframes[0] is train
    assert passed_dataframes[1] is valid
    assert passed_dataframes[2] is test
    np.testing.assert_array_equal(result.train_scores, train["score"].to_numpy())
    np.testing.assert_array_equal(result.validation_scores, valid["score"].to_numpy())
    np.testing.assert_array_equal(result.test_scores, test["score"].to_numpy())


def test_preprocess_split_extracts_wild_types_from_variant_first_letter():
    data = pd.DataFrame(
        {
            "aa_seq": ["ACDEFGHIKL", "MNPQRSTVWY"],
            "variant": ["A1G", "N6C"],
        }
    )

    with patch(
        "megascale.data_processing.preprocessing.construct_list_of_sequence_environments",
        return_value=["dummy_env_1", "dummy_env_2"],
    ), patch(
        "megascale.data_processing.preprocessing.construct_feature_matrix",
        return_value=np.zeros((2, 11, 40)),
    ) as mock_feature_matrix:
        _preprocess_split(data, "one-hot")

    passed_wild_types = mock_feature_matrix.call_args.args[1]
    assert passed_wild_types == ["A", "N"]
