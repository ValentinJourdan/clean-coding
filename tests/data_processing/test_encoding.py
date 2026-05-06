import numpy as np

from megascale.data_processing.encoding import construct_feature_matrix, ENCODINGS


def test_construct_feature_matrix_shape():
    sequences = ["ACD", "GHI"]
    wild_types = ["A", "G"]
    encoding = ENCODINGS["one-hot"]
    encoding_dim = len(next(iter(encoding.values())))

    result = construct_feature_matrix(sequences, wild_types, encoding)

    assert result.shape == (2, 3, 2 * encoding_dim)


def test_construct_feature_matrix_wt_and_mutant_differ_at_center():
    sequences = ["ACA"]
    wild_types = ["G"]
    encoding = ENCODINGS["one-hot"]

    result = construct_feature_matrix(sequences, wild_types, encoding)

    center_idx = 1
    encoding_dim = len(next(iter(encoding.values())))
    wt_center = result[0, center_idx, :encoding_dim]
    mut_center = result[0, center_idx, encoding_dim:]
    assert not np.array_equal(wt_center, mut_center)
