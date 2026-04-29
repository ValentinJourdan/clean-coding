import random

import pandas as pd


def get_data_splitting(
    data: pd.DataFrame, random_seed_int: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, float]:
    """Splits data into train/validation/test sets.

    Args:
        data: Full data.
        random_seed_int: Seed to reproduce the same random subsets.

    Returns:
        Three datasets that correspond to the train/validation/test sets.
        The training set proportion w.r.t. the full dataset.

    """
    pdb_id_list = list(data["pdbid"].unique())

    # Set random seed and shuffle data
    random.seed(random_seed_int)
    random.shuffle(pdb_id_list)

    # 30 test and 30 validation PDBs, remaining ones are for training
    test_pdb_list = pdb_id_list[:30]
    validation_pdb_list = pdb_id_list[30:60]
    train_pdb_list = pdb_id_list[60:]

    train = data[data["pdbid"].isin(train_pdb_list)]
    valid = data[data["pdbid"].isin(validation_pdb_list)]
    test = data[data["pdbid"].isin(test_pdb_list)]

    train_pr = len(train) / (len(train) + len(valid) + len(test))

    return train, valid, test, train_pr
