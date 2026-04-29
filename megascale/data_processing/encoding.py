import numpy as np

PADDING_AMINO_ACID_LETTER = "X"
AMINO_ACIDS_SORTED = [
    "A",
    "C",
    "D",
    "E",
    "F",
    "G",
    "H",
    "I",
    "K",
    "L",
    "M",
    "N",
    "P",
    "Q",
    "R",
    "S",
    "T",
    "V",
    "W",
    "Y",
]

_ONE_HOT_ENCODINGS: dict[str, np.ndarray] = {
    **dict(zip(AMINO_ACIDS_SORTED, np.eye(len(AMINO_ACIDS_SORTED)))),
    PADDING_AMINO_ACID_LETTER: np.zeros(len(AMINO_ACIDS_SORTED)),
}

# Data obtained from Table 3 of https://doi.org/10.1093/mp/sst148
_Z_SCALES: dict[str, np.ndarray] = {
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
    PADDING_AMINO_ACID_LETTER: np.array([0.0, 0.0, 0.0]),
}

ENCODINGS: dict[str, dict[str, np.ndarray]] = {
    "one-hot": _ONE_HOT_ENCODINGS,
    "zscales": _Z_SCALES,
}


def construct_feature_matrix(
    aa_sequences_mutated: list,
    wild_types: list,
    encodings: dict[str, np.ndarray],
) -> np.ndarray:
    """Constructs a feature matrix for amino acid sequences.

    The encoding for each amino acid in the mutated sequence is concatenated to
    the one from the wild type, producing a feature vector per position.

    Args:
        aa_sequences_mutated: A list of sequences, each one represented as a string of
                              amino acid representing letters. The sequences are the
                              ones of the mutated sequence, with the mutation position
                              at the center of the sequence.
        wild_types: List of letters, representing the amino acids that are present
                    in the place of the mutation (at center of sequence) in the wild
                    type protein.
        encodings: Mapping from amino acid letter to its encoding vector.

    Returns:
        The feature matrix of shape (number of sequences, seq_length, 2 * encoding_dim).

    """
    seq_length = len(aa_sequences_mutated[0])
    center_idx = seq_length // 2
    encoding_dim = len(next(iter(encodings.values())))

    feature_matrix = np.zeros(
        shape=(len(aa_sequences_mutated), seq_length, 2 * encoding_dim)
    )

    for seq_idx, (aa_sequence, wt) in enumerate(zip(aa_sequences_mutated, wild_types)):
        features_mutant = np.vstack([encodings[aa] for aa in aa_sequence])
        aa_sequence_wt = aa_sequence[:center_idx] + wt + aa_sequence[center_idx + 1 :]
        features_wt = np.vstack([encodings[aa] for aa in aa_sequence_wt])
        feature_matrix[seq_idx] = np.hstack([features_wt, features_mutant])

    return feature_matrix
