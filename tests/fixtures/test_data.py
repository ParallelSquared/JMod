"""
Test data fixtures for JMod tests
"""
import numpy as np

# Sample spectral library entry
SAMPLE_LIBRARY_ENTRY = {
    "mod_seq": "PEPTIDE",
    "prec_mz": 400.2145,
    "prec_z": 2,
    "iRT": 45.3,
    "IonMob": 0.95,
    "frags": {
        "b2_1": [227.1026, 1000.0],
        "b3_1": [324.1554, 800.0],
        "b4_1": [425.2031, 600.0],
        "y2_1": [276.1555, 900.0],
        "y3_1": [377.2032, 700.0],
        "y4_1": [490.2872, 500.0],
    },
    "spectrum": np.array([
        [227.1026, 1000.0],
        [276.1555, 900.0],
        [324.1554, 800.0],
        [377.2032, 700.0],
        [425.2031, 600.0],
        [490.2872, 500.0],
    ])
}

# Sample mTRAQ modifications
SAMPLE_MTRAQ_TAGS = {
    "mTRAQ-0": 140.0949630177,
    "mTRAQ-4": 144.1020624177,
    "mTRAQ-8": 148.1091618309,
}

# Sample DIANN modifications
SAMPLE_DIANN_MODS = {
    "Carbamidomethyl": 57.021464,
    "Oxidation": 15.994915,
    "Phospho": 79.966331,
}

# Sample MS/MS spectrum
SAMPLE_MS2_SPECTRUM = {
    "scan_num": 1000,
    "RT": 30.5,
    "prec_mz": 500.25,
    "ms1window": [495.0, 505.0],
    "peaks": np.array([
        [147.1128, 1000.0],
        [227.1026, 800.0],
        [276.1555, 900.0],
        [324.1554, 600.0],
        [377.2032, 700.0],
        [425.2031, 400.0],
        [490.2872, 500.0],
        [588.3400, 300.0],
    ])
}

# Sample feature from Dinosaur/Biosaur
SAMPLE_FEATURE = {
    "mz": 500.2501,
    "rtStart": 29.8,
    "rtApex": 30.5,
    "rtEnd": 31.2,
    "intensity": 1e6,
    "charge": 2,
}

# Complex peptide sequences for testing
COMPLEX_PEPTIDES = {
    "phospho": "PEPS(+79.97)TIDE",
    "oxidation": "PEPTM(+15.99)IDE",
    "multiple_mods": "P(+15.99)EPS(+79.97)TM(+15.99)IDE",
    "mtraq_single": "K(mTRAQ-0)PEPTIDE",
    "mtraq_multiple": "K(mTRAQ-0)PEPTIDEK(mTRAQ-0)",
    "mixed_mods": "K(mTRAQ-0)PEPS(+79.97)TIDE",
}

# Expected results for change_seq tests
CHANGE_SEQ_EXPECTED = {
    "PEPTIDE": {
        "diann": "LDLTSED",
        "rev": "DITPEPE"
    },
    "ACDEFGHIKLMNPQRSTVWY": {
        "diann": "LSEDLLSVLVLQLNLTSLLS",
        "rev": "YWVTSRQPNMLKIHGFEDCA"
    },
}

# Sample RT alignment data
RT_ALIGNMENT_DATA = {
    "library_rts": np.array([10.0, 20.0, 30.0, 40.0, 50.0]),
    "observed_rts": np.array([10.5, 20.8, 31.2, 41.5, 52.0]),
    "expected_slope": 1.04,
    "expected_intercept": 0.3,
}