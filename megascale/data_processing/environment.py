PADDING_AMINO_ACID_LETTER = "X"


def _extract_sequence_environment(
    aa_seq: str, center_idx: int, steps_in_one_direction: int
) -> str:
    if len(aa_seq) == 0:
        raise RuntimeError(
            "Pass an amino acid sequence containing at least one amino acid."
        )

    padding = PADDING_AMINO_ACID_LETTER * steps_in_one_direction
    padded_seq = padding + aa_seq + padding

    return padded_seq[center_idx : center_idx + 2 * steps_in_one_direction + 1]


def _mutation_string_to_position_number(mutation: str) -> int:
    extracted_digits = "".join(char for char in mutation if char.isdigit())
    return int(extracted_digits)


def construct_list_of_sequence_environments(
    aa_seqs: list, mutations: list, steps_in_one_direction: int
) -> list:
    """Constructs a list of sequence environments around a mutated residue.

    Environments are padded with the letter X if mutation position is close to start
    or end of original full sequence.

    Args:
        aa_seqs: The full amino acid sequences.
        mutations: The description of the mutation, e.g., W1D or N6C.
        steps_in_one_direction: Number of steps into each direction to include in
                                the environment.

    Returns:
        The shorter sequence substrings of length (2 * steps_in_one_direction + 1)
        representing the sequence environments around a mutated residue.

    """
    seq_environments = []
    for seq, mut in zip(aa_seqs, mutations):
        mutation_idx = _mutation_string_to_position_number(mut) - 1
        seq_environments.append(
            _extract_sequence_environment(seq, mutation_idx, steps_in_one_direction)
        )

    return seq_environments
