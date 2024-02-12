import functools

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


@functools.cache
def _construct_one_hot_encoding_vector_for_each_amino_acid() -> dict[str, np.ndarray]:
    encoding = {}
    num_amino_acids = len(AMINO_ACIDS_SORTED)
    for idx, amino_acid in enumerate(AMINO_ACIDS_SORTED):
        vec = np.zeros((num_amino_acids,))
        vec[idx] = 1
        encoding[amino_acid] = vec

    # for padding
    encoding[PADDING_AMINO_ACID_LETTER] = np.zeros((num_amino_acids,))

    return encoding


def construct_feature_matrix_with_one_hot_encoding(
    aa_sequences_mutated: list, wild_types: list
) -> np.ndarray:
    """Constructs a feature matrix for amino acid sequences.

    The embedding for each amino acid is the simple one-hot encoding. The encoding
    for the amino acid in the mutated sequence is concatenated to the one from the
    wild type.

    Args:
        aa_sequences_mutated: A list of sequences, each one represented as a string of
                              amino acid representing letters. The sequences are the
                              ones of the mutated sequence, with the mutation position
                              at the center of the sequence.
        wild_types: List of letters, representing the amino acids that are present
                    in the place of the mutation (at center of sequence) in the wild
                    type protein.

    Returns:
        The feature matrix of shape (number of sequences, 40).

    """
    aa_encodings = _construct_one_hot_encoding_vector_for_each_amino_acid()
    seq_length = len(aa_sequences_mutated[0])
    center_idx = seq_length // 2

    feature_matrix = np.zeros(shape=(len(aa_sequences_mutated), seq_length, 40))

    for seq_idx, (aa_sequence, wt) in enumerate(zip(aa_sequences_mutated, wild_types)):
        features_mutant = np.vstack(
            [aa_encodings[amino_acid] for amino_acid in aa_sequence]
        )
        aa_sequence_wt = aa_sequence[:center_idx] + wt + aa_sequence[center_idx + 1 :]
        features_wt = np.vstack(
            [aa_encodings[amino_acid] for amino_acid in aa_sequence_wt]
        )
        feature_matrix[seq_idx] = np.hstack([features_wt, features_mutant])

    return feature_matrix
