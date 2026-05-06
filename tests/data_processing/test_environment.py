from megascale.data_processing.environment import (
    construct_list_of_sequence_environments,
    _mutation_string_to_position_number,
)


def test_mutation_string_to_position_number():
    assert _mutation_string_to_position_number("W3A") == 3
    assert _mutation_string_to_position_number("N12C") == 12


def test_construct_list_of_sequence_environments_length():
    seqs = ["ACDEFGHIK"]
    mutations = ["A1G"]
    steps = 2

    result = construct_list_of_sequence_environments(seqs, mutations, steps)

    assert len(result) == 1
    assert len(result[0]) == 2 * steps + 1
