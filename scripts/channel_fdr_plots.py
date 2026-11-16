#!/usr/bin/env python
"""The channel-FDR plots of a run, from a written channel feature table, into
the table's folder (src.channel_fdr_plots.plot_channel_fdr): the best channel
by PredVal, IDs vs Channel_Qvalue, the IDs retained at the FDR threshold, the
feature importance if the fit wrote it beside the table, and every feature's
ECDF (into ECDF/).

    python scripts/channel_fdr_plots.py <run>/channel_fdr/channel_features.parquet
    python scripts/channel_fdr_plots.py <run>/channel_fdr/channel_features.parquet --split KR
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("features", help="the run's channel_features.parquet")
    p.add_argument("--split", choices=["all", "KR"], default="all",
                   help="every precursor together, or the K- and R-ending peptides side by side")
    p.add_argument("--workers", type=int, default=os.cpu_count() or 1, help="processes drawing the ECDFs")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    # JMod's config parses the command line when it is imported, and these are not its options
    sys.argv[1:] = []
    import polars as pl
    from src.channel_fdr_plots import plot_channel_fdr, read_feature_importance

    folder = os.path.dirname(os.path.abspath(args.features))
    plot_channel_fdr(pl.read_parquet(args.features), folder, importance=read_feature_importance(folder),
                     split=args.split, workers=args.workers)
