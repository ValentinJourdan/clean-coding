import pandas as pd

from megascale.data_processing.split import get_data_splitting, NUM_TEST_PDBS, NUM_VALIDATION_PDBS


def _make_data_with_n_pdbs(n: int) -> pd.DataFrame:
    rows = [{"pdbid": f"pdb{i}", "score": 0.0} for i in range(n)]
    return pd.DataFrame(rows)


def test_splits_cover_all_data():
    n_pdbs = NUM_TEST_PDBS + NUM_VALIDATION_PDBS + 10
    data = _make_data_with_n_pdbs(n_pdbs)

    train, valid, test, _ = get_data_splitting(data, random_seed_int=42)

    assert len(train) + len(valid) + len(test) == len(data)


def test_splits_are_disjoint():
    n_pdbs = NUM_TEST_PDBS + NUM_VALIDATION_PDBS + 10
    data = _make_data_with_n_pdbs(n_pdbs)

    train, valid, test, _ = get_data_splitting(data, random_seed_int=42)

    train_ids = set(train["pdbid"])
    valid_ids = set(valid["pdbid"])
    test_ids = set(test["pdbid"])
    assert train_ids.isdisjoint(valid_ids)
    assert train_ids.isdisjoint(test_ids)
    assert valid_ids.isdisjoint(test_ids)
