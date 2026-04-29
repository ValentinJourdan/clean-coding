import random

import pandas as pd

NUM_TEST_PDBS = 30
NUM_VALIDATION_PDBS = 30


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

    random.seed(random_seed_int)
    random.shuffle(pdb_id_list)

    test_pdb_list = pdb_id_list[:NUM_TEST_PDBS]
    validation_pdb_list = pdb_id_list[
        NUM_TEST_PDBS : NUM_TEST_PDBS + NUM_VALIDATION_PDBS
    ]
    train_pdb_list = pdb_id_list[NUM_TEST_PDBS + NUM_VALIDATION_PDBS :]

    train = data[data["pdbid"].isin(train_pdb_list)]
    valid = data[data["pdbid"].isin(validation_pdb_list)]
    test = data[data["pdbid"].isin(test_pdb_list)]

    train_pr = len(train) / (len(train) + len(valid) + len(test))

    return train, valid, test, train_pr
