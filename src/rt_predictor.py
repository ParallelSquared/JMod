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

"""Retention-time and ion-mobility prediction, backed by JRIF.

Replaces the TensorFlow ensemble in the old ``finetune_funs``: one PyTorch
model per run, calibrated on the run's own identifications.

* **Modified sequences.**  The old encoder saw 20 plain amino acids, so a
  labeled peptide and its unlabeled counterpart were the same input.  JRIF
  reads modifications, and accepts JMod's ``(UniMod:N)`` notation as is; only
  the mass-tag channels need translating (:func:`to_jrif_seq`).

* **Calibration.**  ``calibrate()`` subsumes both halves of the old design --
  the LOWESS ``obs_to_model``/``model_to_obs`` pair is JRIF's isotonic warp,
  and the ``FT_minimum`` ID-count threshold is replaced by a measured
  comparison of warp-only against warp-plus-fine-tuning.  After calibration,
  :meth:`JmodRTPredictor.predict_seqs` returns RT in the observed units of the
  data it was calibrated on.

* **Ion mobility** (:class:`JmodIMPredictor`) is the same design keyed on
  (sequence, charge); rt_alignment decides whether it or the library's own IM
  is used for the run.

One guard here works around JRIF rather than JMod, marked ``BANDAID U5``
against JRIF's ``KNOWN_ISSUES.md``.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from src.logger import logger
from src.utils.errors import JModError
from src.utils.parse_peptides import split_nterm

# Tag name (substring of massTag.name) -> chemistry suffix of the JRIF
# checkpoint names, "{kind}_{chemistry}" with kind "rt" or "im".  Checkpoints
# that do not exist yet resolve to the label-free "{kind}_lf"; drop the .pt
# into JRIF and it is picked up here with no change to this file.
_TAG_CHEMISTRIES = {
    "PSMtag": "psmtag",
    "mTRAQ": "mtraq",
    "diethyl": "diethyl",
    "dimethyl": "dimethyl",
}
_LABEL_FREE = "lf"

# Device for JRIF calibration (training).  CPU, because training on the GPU is
# not reproducible: on Apple MPS two calibrations on identical input in fresh
# processes differ (RT predictions by up to 0.03 s, IM by 5e-5 1/K0), even with
# torch.use_deterministic_algorithms; on CPU they are bitwise identical.  The
# drift moved the fitted RT tolerance, hence the MS1 tolerance, the MS2
# tolerance set from it, and the IDs, between otherwise identical runs.  Costs
# ~8 s per model on this Mac (RT 5 -> 13 s, IM 7 -> 14 s for ~4,000 IDs).
# Prediction keeps JRIF's default device: for fixed weights it is reproducible
# on MPS (tested across processes).
_CALIBRATION_DEVICE = "cpu"


def to_jrif_seq(mod_seq: str, tag=None) -> str:
    """Translate a JMod modified sequence into JRIF's notation.

    JRIF reads JMod's ``(UniMod:N)`` parentheticals directly; what it cannot
    read is JMod's mass-tag channels, ``(PSMtag_5plex-4)``.  Those become a
    delta at the tag's *base* mass, ``K[+308.1161]`` on a residue and
    ``n[+308.1161]`` on the N-terminus.  The channel is discarded on purpose:
    channels are isotopologues that co-elute, so every channel is one token --
    which is also how the pretrained vocabularies were built.

    Anything else passes through untouched, so an annotation JRIF cannot
    represent (another tag's channel, an unknown accession) reaches it intact
    and is refused there rather than silently rewritten here.

    >>> to_jrif_seq("AAAGELQEDSGLC(UniMod:4)VLAR")
    'AAAGELQEDSGLC(UniMod:4)VLAR'
    """
    if not isinstance(mod_seq, str) or tag is None:
        return mod_seq
    channel = re.compile(rf"\({re.escape(tag.name)}-\w+\)")
    delta = f"[{float(tag.mass):+.4f}]"
    nterm, body = split_nterm(mod_seq)
    nterm = channel.sub("n" + delta, nterm)
    return nterm + channel.sub(delta, body)


def _shipped(name: str) -> Optional[Path]:
    """Path of a JRIF checkpoint that actually exists, else None."""
    from jrif.checkpoints import checkpoint_path

    try:
        path = Path(checkpoint_path(name))
    except KeyError:
        return None
    # checkpoint_path does not check that the file exists, so a missing one
    # would otherwise surface as an opaque torch.load failure.
    return path if path.exists() else None


def checkpoint_for_tag(tag, kind: str = "rt") -> tuple[Path, str]:
    """The JRIF *kind* ("rt" or "im") checkpoint for *tag*'s chemistry.

    Returns ``(path, name)``.  A tag with no dedicated base model falls back to
    the label-free base: ``calibrate`` extends the vocabulary with the label's
    tokens and learns them from the run's own identifications.  It is weaker
    than a dedicated base, so it is logged.
    """
    chemistry = next(
        (chem for frag, chem in _TAG_CHEMISTRIES.items() if tag is not None and frag in tag.name),
        None,
    )

    if chemistry is not None:
        name = f"{kind}_{chemistry}"
        path = _shipped(name)
        if path is not None:
            return path, name
        logger.info(
            f"No {kind.upper()} model trained for {tag.name}; using the label-free base "
            f"and learning the label from this run's IDs"
        )

    name = f"{kind}_{_LABEL_FREE}"
    path = _shipped(name)
    if path is not None:
        return path, name

    # TEMPORARY until JRIF ships im_lf: the only IM base is im_psmtag.  An
    # unlabeled peptide is representable in its vocabulary (<nterm>, plain K),
    # and calibrate() adapts it to this run, but it was trained on labeled
    # chemistry.  Delete this branch when im_lf.pt lands.
    if kind == "im":
        logger.warning(
            "This JRIF has no label-free IM model (im_lf); calibrating the "
            "PSMtag IM model (im_psmtag) on this run instead"
        )
        name = "im_psmtag"
        path = _shipped(name)
        if path is not None:
            return path, name

    raise JModError(f"JRIF ships no {kind.upper()} checkpoint usable for "
                    f"{tag.name if tag is not None else 'label-free'} data")


def _predict_representable(predict, frame: pd.DataFrame, vocab, what: str) -> np.ndarray:
    """``predict(frame)``, with NaN for rows JRIF cannot represent.

    A sequence the model cannot represent -- typically a library modification
    no identification carried, so calibration never learned it -- gets NaN,
    and the count is logged.  JRIF refuses such rows rather than guessing; NaN
    keeps that refusal visible downstream instead of failing the whole library
    on one peptide.
    """
    from jrif import TokenizationError

    out = np.full(len(frame), np.nan)
    try:
        out[:] = predict(frame)
        return out
    except TokenizationError:
        pass

    from jrif.dataset import tokenize

    bad, first = set(), None
    for s in dict.fromkeys(frame["seq"]):
        try:
            tokenize(s, vocab)
        except TokenizationError as e:
            bad.add(s)
            first = first or str(e)
    ok = ~frame["seq"].isin(bad).to_numpy()
    if ok.any():
        out[ok] = predict(frame[ok].reset_index(drop=True))
    logger.warning(
        f"{int((~ok).sum())} of {len(frame)} peptides cannot be represented by the "
        f"{what} model and have no predicted {what} (first: {first})"
    )
    return out


class JmodRTPredictor:
    """A JRIF predictor calibrated to one run, speaking JMod's sequence notation."""

    def __init__(self, predictor, tag=None, report: Optional[dict] = None):
        self._predictor = predictor
        self._tag = tag
        self.report = report or {}

    @classmethod
    def calibrate(
        cls,
        grouped_df: pd.DataFrame,
        *,
        tag=None,
        seq_col: str = "mod_seq",
        rt_col: str = "RT",
        holdout_fraction: float = 0.2,
        seed: int = 42,
    ) -> tuple["JmodRTPredictor", pd.DataFrame]:
        """Calibrate on this run's identifications.

        Returns ``(predictor, holdout)``.  BANDAID U5: the holdout is split off
        here, never shown to JRIF.  JRIF's own holdout drives its mode and
        epoch selection, so the error it reports is a selection score, not an
        honest one; the empirical-vs-predicted comparison in rt_alignment needs
        rows the model never saw.
        """
        from jrif import RTPredictor

        ckpt, ckpt_name = checkpoint_for_tag(tag)
        logger.info(f"{len(grouped_df)} peptides considered for RT calibration ({ckpt_name})")

        work = grouped_df.dropna(subset=[seq_col, rt_col]).drop_duplicates(subset=[seq_col])
        holdout = work.sample(frac=holdout_fraction, random_state=seed)
        train = work.drop(holdout.index)

        predictor = RTPredictor.from_checkpoint(ckpt)
        # IDs JRIF cannot represent are skipped there, with one printed line.
        report = predictor.calibrate(
            pd.DataFrame({
                "seq": [to_jrif_seq(s, tag) for s in train[seq_col]],
                "rt": train[rt_col].to_numpy(),
            }),
            seq_col="seq",
            rt_col="rt",
            device=_CALIBRATION_DEVICE,
            verbose=False,
        )
        logger.info(
            f"RT calibration: mode={report.get('mode')} "
            f"n={report.get('n')} holdout_mae={report.get('holdout_mae'):.3f}"
        )
        if report.get("mode") == "finetune_raw":
            logger.warning(
                "The RT base model could not rank this run's own IDs "
                f"(Spearman {report.get('holdout_rank'):.3f}); it was fine-tuned from "
                "raw targets instead of calibrated.  Check that the label and the "
                "chosen base model match."
            )

        return cls(predictor, tag=tag, report=report), holdout

    def qc_plot(self, holdout: pd.DataFrame, results_folder: Optional[str],
                seq_col: str = "mod_seq", rt_col: str = "RT") -> None:
        """Observed against predicted RT on the held-out identifications.

        The old ensemble drew five plots, four of which described the LOWESS
        calibration that JRIF now does internally.  What is left is the one
        that actually says whether the model works, and it is drawn on rows
        the model never trained on.
        """
        if not results_folder:
            return
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            observed = holdout[rt_col].to_numpy()
            predicted = self.predict_seqs(holdout[seq_col].tolist())
            ok = np.isfinite(predicted)
            observed, predicted = observed[ok], predicted[ok]
            mae = float(np.abs(predicted - observed).mean())

            plt.subplots()
            plt.scatter(observed, predicted, s=1)
            lim = [min(observed.min(), predicted.min()), max(observed.max(), predicted.max())]
            plt.plot(lim, lim, color="tab:orange", linewidth=1)
            plt.xlabel("Observed RT (held out)")
            plt.ylabel("Predicted RT")
            plt.title(f"{self.report.get('mode', 'calibrated')}  n={len(observed)}  MAE={mae:.3f}")
            plt.savefig(
                Path(results_folder) / "first_search" / "fine_tuning" / "RT_finetune_ObsvsPred.png",
                dpi=600, bbox_inches="tight",
            )
            plt.close()
        except Exception as e:
            logger.warning(f"Could not draw the RT calibration QC plot: {type(e).__name__}: {e}")

    def predict_seqs(self, mod_seqs: Sequence[str]) -> np.ndarray:
        """Predicted RT for each sequence, in the calibration data's units.

        Unrepresentable sequences get NaN (see :func:`_predict_representable`).
        """
        mod_seqs = list(mod_seqs)
        if not mod_seqs:
            return np.empty(0, dtype=float)

        # Translate each distinct sequence once; JRIF dedupes again internally.
        unique = list(dict.fromkeys(mod_seqs))
        frame = pd.DataFrame({"seq": [to_jrif_seq(s, self._tag) for s in unique]})
        rts = _predict_representable(
            lambda f: self._predictor.predict(f, seq_col="seq"),
            frame, self._predictor.vocab, "RT",
        )
        lookup = dict(zip(unique, rts))
        return np.array([lookup[s] for s in mod_seqs], dtype=float)


class JmodIMPredictor:
    """A JRIF IM predictor calibrated to one run, speaking JMod's sequence notation.

    Unlike RT there is no holdout here: the caller splits the anchors once and
    scores this model and the library-IM alignment on the same held-out rows.
    """

    def __init__(self, predictor, tag=None, report: Optional[dict] = None):
        self._predictor = predictor
        self._tag = tag
        self.report = report or {}

    @classmethod
    def calibrate(
        cls,
        train: pd.DataFrame,
        *,
        tag=None,
        seq_col: str = "mod_seq",
        charge_col: str = "z",
        im_col: str = "im",
    ) -> "JmodIMPredictor":
        """Calibrate on this run's identifications (observed 1/K0 per peptide ion)."""
        from jrif import IMPredictor

        ckpt, ckpt_name = checkpoint_for_tag(tag, kind="im")
        logger.info(f"{len(train)} peptide ions considered for IM calibration ({ckpt_name})")

        predictor = IMPredictor.from_checkpoint(ckpt)
        # IDs JRIF cannot represent are skipped there, with one printed line.
        report = predictor.calibrate(
            pd.DataFrame({
                "seq": [to_jrif_seq(s, tag) for s in train[seq_col]],
                "charge": train[charge_col].to_numpy(dtype=float),
                "im": train[im_col].to_numpy(dtype=float),
            }),
            seq_col="seq",
            charge_col="charge",
            im_col="im",
            device=_CALIBRATION_DEVICE,
            verbose=False,
        )
        logger.info(
            f"IM calibration: mode={report.get('mode')} "
            f"n={report.get('n')} holdout_mae={report.get('holdout_mae'):.4f}"
        )
        if report.get("mode") == "finetune_raw":
            logger.warning(
                "The IM base model could not rank this run's own IDs "
                f"(Spearman {report.get('holdout_rank'):.3f}); it was fine-tuned from "
                "raw targets instead of calibrated.  Check that the label and the "
                "chosen base model match."
            )
        return cls(predictor, tag=tag, report=report)

    def predict(self, mod_seqs: Sequence[str], charges: Sequence[float]) -> np.ndarray:
        """Predicted 1/K0 for each (sequence, charge), in the calibration data's units.

        Unrepresentable sequences get NaN (see :func:`_predict_representable`).
        """
        mod_seqs = list(mod_seqs)
        charges = np.asarray(charges, dtype=float)
        if len(mod_seqs) != len(charges):
            raise ValueError(f"{len(mod_seqs)} sequences but {len(charges)} charges")
        if not mod_seqs:
            return np.empty(0, dtype=float)

        # Translate each distinct sequence once; JRIF dedupes (seq, charge) itself.
        translated = {s: to_jrif_seq(s, self._tag) for s in dict.fromkeys(mod_seqs)}
        frame = pd.DataFrame({"seq": [translated[s] for s in mod_seqs], "charge": charges})
        return _predict_representable(
            lambda f: self._predictor.predict(f, seq_col="seq", charge_col="charge"),
            frame, self._predictor.vocab, "IM",
        )

    def save(self, path) -> None:
        self._predictor.save(path)
