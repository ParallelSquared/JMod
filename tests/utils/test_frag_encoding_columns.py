"""encode_frag_columns must agree with encode_frag_name everywhere."""
import numpy as np
import pytest

from src.utils.frag_encoding import (
    encode_frag_name,
    encode_frag_columns,
    _ION_TYPE_TO_INT,
    _LOSS_TO_INT,
)


def build_name(ion, idx, loss, charge, iso):
    name = f"{ion}{idx}"
    if loss:
        name += f"-{loss}"
    name += f"_{charge}"
    if iso:
        name += f"_iso{iso}"
    return name


def test_matches_string_encoder_across_full_range():
    ions = sorted(_ION_TYPE_TO_INT)
    # known losses plus one unknown, which both encoders map to 0 (no loss)
    losses = sorted(_LOSS_TO_INT) + ["CO"]
    indices = list(range(1, 256))
    charges = list(range(1, 9))
    isos = [0, 1, 7, 15]

    grid = [
        (ion, idx, loss, charge, iso)
        for ion in ions
        for idx in indices
        for loss in losses
        for charge in charges
        for iso in isos
    ]
    expected = np.array(
        [encode_frag_name(build_name(*combo)) for combo in grid], dtype=np.int32
    )
    actual = encode_frag_columns(
        [c[0] for c in grid],
        [c[1] for c in grid],
        [c[2] for c in grid],
        [c[3] for c in grid],
        [c[4] for c in grid],
    )
    assert actual.dtype == np.int32
    assert np.array_equal(actual, expected)


def test_iso_defaults_to_zero():
    with_default = encode_frag_columns(["y"], [4], ["H2O"], [2])
    explicit = encode_frag_columns(["y"], [4], ["H2O"], [2], [0])
    assert np.array_equal(with_default, explicit)
    assert with_default[0] == encode_frag_name("y4-H2O_2")


def test_unknown_ion_type_raises():
    with pytest.raises(ValueError, match="ion type"):
        encode_frag_columns(["q"], [1], [""], [1])


def test_empty_input():
    out = encode_frag_columns([], [], [], [])
    assert out.dtype == np.int32 and len(out) == 0
