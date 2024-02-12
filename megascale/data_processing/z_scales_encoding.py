import numpy as np

# Data obtained from Table 3 of https://doi.org/10.1093/mp/sst148
Z_SCALES = {
    "F": np.array([-4.92, 1.3, 0.45]),
    "W": np.array([-4.75, 3.65, 0.85]),
    "I": np.array([-4.44, -1.68, -1.03]),
    "L": np.array([-4.19, -1.03, -0.98]),
    "V": np.array([-2.69, -2.53, -1.29]),
    "M": np.array([-2.49, -0.27, -0.41]),
    "Y": np.array([-1.39, 2.32, 0.01]),
    "P": np.array([-1.22, 0.88, 2.23]),
    "A": np.array([0.07, -1.73, 0.09]),
    "C": np.array([0.71, -0.97, 4.13]),
    "T": np.array([0.92, -2.09, -1.4]),
    "S": np.array([1.96, -1.63, 0.57]),
    "Q": np.array([2.19, 0.53, -1.14]),
    "G": np.array([2.23, -5.36, 0.3]),
    "H": np.array([2.41, 1.74, 1.11]),
    "K": np.array([2.84, 1.41, -3.14]),
    "R": np.array([2.88, 2.52, -3.44]),
    "E": np.array([3.08, 0.039, -0.07]),
    "N": np.array([3.22, 1.45, 0.84]),
    "D": np.array([3.64, 1.13, 2.36]),
    "X": np.array([0.0, 0.0, 0.0]),  # for padding
}


def construct_feature_matrix_with_z_scales_encoding(
    aa_sequences_mutated: list, wild_types: list
) -> np.ndarray:
    """Constructs a feature matrix for amino acid sequences.

    The embedding for each amino acid are three numbers obtained from a principal
    component analysis of 29 physicochemical properties for the amino acids (see
    https://doi.org/10.1093/mp/sst148). The encoding for the amino acid in the
    mutated sequence is concatenated to the one from the wild type.

    Args:
        aa_sequences_mutated: A list of sequences, each one represented as a string of
                              amino acid representing letters. The sequences are the
                              ones of the mutated sequence, with the mutation position
                              at the center of the sequence.
        wild_types: List of letters, representing the amino acids that are present
                    in the place of the mutation (at center of sequence) in the wild
                    type protein.

    Returns:
        The feature matrix of shape (number of sequences, 6).

    """
    seq_length = len(aa_sequences_mutated[0])
    center_idx = seq_length // 2

    feature_matrix = np.zeros(shape=(len(aa_sequences_mutated), seq_length, 6))

    for seq_idx, (aa_sequence, wt) in enumerate(zip(aa_sequences_mutated, wild_types)):
        features_mutant = np.vstack(
            [Z_SCALES[amino_acid] for amino_acid in aa_sequence]
        )
        aa_sequence_wt = aa_sequence[:center_idx] + wt + aa_sequence[center_idx + 1 :]
        features_wt = np.vstack([Z_SCALES[amino_acid] for amino_acid in aa_sequence_wt])
        feature_matrix[seq_idx] = np.hstack([features_wt, features_mutant])

    return feature_matrix
