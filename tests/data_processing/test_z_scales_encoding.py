from megascale.data_processing.z_scales_encoding import Z_SCALES


def test_z_scales_dict_has_expected_number_of_entries():
    num_expected_amino_acid_z_scales = 21
    assert len(Z_SCALES) == num_expected_amino_acid_z_scales
