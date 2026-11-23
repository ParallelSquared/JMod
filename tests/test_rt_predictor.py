#  Copyright (c) 2026 Parallel Squared Technology Institute
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#          http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

"""JMod's sequence notation reaching JRIF as the tokens we mean.

Assertions go through ``jrif.dataset.tokenize`` to token names, not just the
translated string: the string is an intermediate, the tokens are what the
model sees.
"""

import numpy as np
import pytest

from src.mass_tags import massTag
from src.rt_predictor import JmodRTPredictor, checkpoint_for_tag, to_jrif_seq


@pytest.fixture(scope="module")
def psmtag():
    return massTag("nK", 308.1160923903, [0.0, 4.01095605604, 8.0268387024],
                   ["0", "4", "8"], "PSMtag_5plex")


@pytest.fixture(scope="module")
def mtraq():
    return massTag("nK", 140.0949630177, [0.0, 4.0070994, 8.0141988132],
                   ["0", "4", "8"], "mTRAQ")


def _base(name):
    from jrif import RTPredictor
    from jrif.checkpoints import checkpoint_path

    return RTPredictor.from_checkpoint(checkpoint_path(name))


@pytest.fixture(scope="module")
def psmtag_vocab():
    return _base("rt_psmtag").vocab


def tokens_of(seq, vocab):
    """The non-pad tokens JRIF actually sees, termini included."""
    from jrif.dataset import tokenize

    inverse = {i: t for t, i in vocab.tokens.items()}
    return [inverse[i] for i in tokenize(seq, vocab) if i != 0]


class TestTranslation:
    """Only the mass-tag channels are rewritten; everything else passes through."""

    def test_unlabeled_sequences_pass_through(self, psmtag):
        for s in ("AAAAAAALQAK", "AAAGELQEDSGLC(UniMod:4)VLAR", "PEPM(UniMod:35)IDEK"):
            assert to_jrif_seq(s, psmtag) == s
            assert to_jrif_seq(s) == s

    def test_residue_tag_becomes_a_base_mass_delta(self, psmtag):
        assert to_jrif_seq("PEPTIDEK(PSMtag_5plex-0)", psmtag) == "PEPTIDEK[+308.1161]"

    def test_every_channel_collapses_to_one_token(self, psmtag):
        # Channels are isotopologues that co-elute; one token, one embedding.
        assert {to_jrif_seq(f"PEPTIDEK(PSMtag_5plex-{c})", psmtag) for c in ("0", "4", "8")} == {
            "PEPTIDEK[+308.1161]"
        }

    def test_nterm_tag_becomes_an_nterm_token(self, psmtag):
        assert to_jrif_seq("(PSMtag_5plex-0)PEPTIDEK", psmtag) == "n[+308.1161]PEPTIDEK"

    def test_mtraq_uses_its_own_base_mass(self, mtraq):
        assert to_jrif_seq("(mTRAQ-0)PEPTIDEK(mTRAQ-8)", mtraq) == "n[+140.0950]PEPTIDEK[+140.0950]"

    def test_another_tags_channel_is_left_for_jrif_to_refuse(self, mtraq):
        # An mTRAQ run must not silently absorb a PSMtag annotation.
        assert to_jrif_seq("PEPTIDEK(PSMtag_5plex-0)", mtraq) == "PEPTIDEK(PSMtag_5plex-0)"


class TestTokenizesAsIntended:
    """The translated string reaches the model as the tokens we expect."""

    def test_labeled_peptide(self, psmtag, psmtag_vocab):
        seq = to_jrif_seq("(PSMtag_5plex-0)C(UniMod:4)SQAPVYGRK(PSMtag_5plex-4)", psmtag)
        assert tokens_of(seq, psmtag_vocab) == [
            "n[+308.1161]", "C[+57.0215]", "S", "Q", "A", "P", "V", "Y", "G", "R",
            "K[+308.1161]", "<cterm>",
        ]

    def test_unlabeled_peptide_has_free_termini(self, psmtag_vocab):
        assert tokens_of("PEPTIDEK", psmtag_vocab) == [
            "<nterm>", "P", "E", "P", "T", "I", "D", "E", "K", "<cterm>",
        ]

    def test_oxidised_methionine(self, psmtag_vocab):
        assert "M[+15.9949]" in tokens_of("PEPM(UniMod:35)IDEK", psmtag_vocab)

    def test_untranslated_channel_is_refused_not_dropped(self, mtraq, psmtag_vocab):
        from jrif import TokenizationError

        with pytest.raises(TokenizationError):
            tokens_of(to_jrif_seq("PEPTIDEK(PSMtag_5plex-0)", mtraq), psmtag_vocab)


class TestCheckpointForTag:
    """Chemistry to base model, including the not-yet-trained ones."""

    def test_psmtag_gets_its_own_base(self, psmtag):
        assert checkpoint_for_tag(psmtag)[1] == "rt_psmtag"

    def test_label_free_gets_the_label_free_base(self):
        assert checkpoint_for_tag(None)[1] == "rt_lf"

    def test_plex_variants_match_by_substring(self):
        tag = massTag("nK", 308.1160923903, [0.0], ["0"], "PSMtag_9plex_D16v2")
        assert checkpoint_for_tag(tag)[1] == "rt_psmtag"

    def test_chemistry_without_a_base_falls_back(self, mtraq):
        """mTRAQ has no checkpoint yet; this must degrade, not raise.

        When rt_mtraq.pt ships, this test flips to asserting "rt_mtraq" and
        nothing else has to change.
        """
        path, name = checkpoint_for_tag(mtraq)
        assert name == "rt_lf"
        assert path.exists()

    def test_an_unrecognised_tag_falls_back_rather_than_raising(self):
        # The old fine_tune_rt raised JModError here, which took K_6C13 down.
        tag = massTag("nK", 6.020129, [0.0], ["0"], "K_6C13")
        assert checkpoint_for_tag(tag)[1] == "rt_lf"


@pytest.fixture(scope="module")
def handle():
    return JmodRTPredictor(_base("rt_lf"))


class TestPredictSeqs:
    """Prediction over a library: order preserved, duplicates identical."""

    def test_one_prediction_per_input_row(self, handle):
        out = handle.predict_seqs(["PEPTIDEK", "AAAGELQEDSGLC(UniMod:4)VLAR", "SAMPLER"])
        assert out.shape == (3,)
        assert np.isfinite(out).all()

    def test_duplicates_get_identical_values_in_input_order(self, handle):
        out = handle.predict_seqs(["PEPTIDEK", "SAMPLER", "PEPTIDEK"])
        assert out[0] == out[2]
        assert out[0] != out[1]

    def test_empty_input(self, handle):
        assert handle.predict_seqs([]).shape == (0,)

    def test_modifications_change_the_prediction(self, handle):
        """The whole point of the port: the model must see the modification."""
        plain, modified = handle.predict_seqs(["PEPTIDEMK", "PEPTIDEM(UniMod:35)K"])
        assert plain != modified

    def test_unrepresentable_peptides_get_nan_and_the_rest_still_predict(self, handle, app_log):
        # rt_lf has no N-terminal PSMtag token; calibration would learn it.
        out = handle.predict_seqs(["PEPTIDEK", "n[+308.1161]PEPTIDEK", "SAMPLER"])
        assert np.isfinite(out[[0, 2]]).all()
        assert np.isnan(out[1])
        assert any("cannot be represented" in r.getMessage() for r in app_log)


class TestIMCheckpoint:
    """Same resolution as RT, keyed on kind="im"."""

    def test_psmtag_gets_its_own_im_base(self, psmtag):
        assert checkpoint_for_tag(psmtag, kind="im")[1] == "im_psmtag"

    def test_label_free_im(self, app_log):
        """TEMPORARY: no im_lf ships yet, so label-free IM calibrates im_psmtag.

        When im_lf.pt lands this flips to asserting "im_lf" and the warning
        goes away.
        """
        path, name = checkpoint_for_tag(None, kind="im")
        assert name == "im_psmtag"
        assert path.exists()
        assert any("im_lf" in r.getMessage() for r in app_log)

    def test_chemistry_without_an_im_base_falls_back(self, mtraq):
        assert checkpoint_for_tag(mtraq, kind="im")[1] == "im_psmtag"


@pytest.fixture(scope="module")
def im_handle():
    from jrif import IMPredictor
    from jrif.checkpoints import checkpoint_path

    from src.rt_predictor import JmodIMPredictor

    return JmodIMPredictor(IMPredictor.from_checkpoint(checkpoint_path("im_psmtag")))


class TestIMPredict:
    def test_one_prediction_per_row(self, im_handle):
        out = im_handle.predict(["PEPTIDEK", "SAMPLER", "PEPTIDEK"], [2, 2, 2])
        assert out.shape == (3,)
        assert np.isfinite(out).all()
        assert out[0] == out[2]

    def test_charge_changes_the_prediction(self, im_handle):
        z2, z3 = im_handle.predict(["PEPTIDEK", "PEPTIDEK"], [2, 3])
        assert z2 != z3

    def test_empty_input(self, im_handle):
        assert im_handle.predict([], []).shape == (0,)

    def test_length_mismatch_raises(self, im_handle):
        with pytest.raises(ValueError):
            im_handle.predict(["PEPTIDEK"], [2, 3])

    def test_unrepresentable_peptides_get_nan(self, im_handle, app_log):
        # im_psmtag has no mTRAQ token
        out = im_handle.predict(["PEPTIDEK", "PEPTIDEK[+140.0950]", "SAMPLER"], [2, 2, 2])
        assert np.isfinite(out[[0, 2]]).all()
        assert np.isnan(out[1])
        assert any("cannot be represented by the IM model" in r.getMessage() for r in app_log)


class TestCalibrationDevice:
    """JRIF calibration trains on CPU: GPU (MPS) training is not reproducible."""

    @staticmethod
    def _capture(monkeypatch, cls_name):
        import jrif
        seen = {}

        class _Stub:
            def calibrate(self, df, **kw):
                seen.update(kw)
                return {"mode": "warp", "n": len(df), "holdout_mae": 0.0}

        monkeypatch.setattr(getattr(jrif, cls_name), "from_checkpoint",
                            classmethod(lambda cls, *a, **k: _Stub()))
        return seen

    def test_rt_calibration_runs_on_cpu(self, monkeypatch):
        import pandas as pd
        seen = self._capture(monkeypatch, "RTPredictor")
        df = pd.DataFrame({"mod_seq": [f"PEPTIDE{'K' * i}R" for i in range(10)],
                           "RT": np.arange(10, dtype=float)})
        JmodRTPredictor.calibrate(df, tag=None)
        assert seen.get("device") == "cpu"

    def test_im_calibration_runs_on_cpu(self, monkeypatch):
        import pandas as pd
        from src.rt_predictor import JmodIMPredictor
        seen = self._capture(monkeypatch, "IMPredictor")
        df = pd.DataFrame({"mod_seq": ["PEPTIDEK", "PEPTIDER"], "z": [2.0, 2.0], "im": [0.9, 0.95]})
        JmodIMPredictor.calibrate(df, tag=None)
        assert seen.get("device") == "cpu"
