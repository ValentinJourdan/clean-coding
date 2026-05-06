import pandas as pd
import pytest

from megascale.data_processing.preprocessing import preprocess_data


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
