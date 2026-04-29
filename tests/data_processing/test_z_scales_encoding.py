from megascale.data_processing.encoding import ENCODINGS


def test_z_scales_dict_has_expected_number_of_entries():
    num_expected_amino_acid_z_scales = 21
    assert len(ENCODINGS["zscales"]) == num_expected_amino_acid_z_scales
