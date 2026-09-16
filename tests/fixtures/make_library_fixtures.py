"""Regenerate the spectral-library fixture files in this directory.

Run from the repo root:

    .venv/bin/python tests/fixtures/make_library_fixtures.py

The TSV and parquet variants of each fixture carry the same content and must
parse to identical stores (see tests/models/spec_lib/test_library_golden.py).
Parquet columns are typed like real DIA-NN output (floats/ints) except Decoy
and IonMobility, which stay strings so the "1.0"/"True" and ""/"0.0" edge
cases survive the round trip.
"""
import csv
import os

import pyarrow as pa
import pyarrow.parquet as pq

HERE = os.path.dirname(os.path.abspath(__file__))

EDGECASE_COLUMNS = [
    "ModifiedPeptide", "StrippedPeptide", "PrecursorCharge", "PrecursorMz",
    "Tr_recalibrated", "IonMobility", "ProteinGroup", "ProteinName",
    "ProteinID", "Genes", "Decoy", "FragmentType", "FragmentNumber",
    "FragmentCharge", "FragmentLossType", "FragmentMz", "RelativeIntensity",
]

# One tuple per fragment row, matching EDGECASE_COLUMNS.
EDGECASE_ROWS = [
    # P1: normal precursor. Includes a neutral-loss fragment, an equal-m/z tie
    # (b2_1 vs y2_1, exercises stable spectrum sort), and a duplicate y4_1 key
    # where the later row must win.
    ("PEPTIDEK", "PEPTIDEK", "2", "464.75", "25.5", "0.95", "PG1", "ProtA", "P001", "GENE1", "", "y", "4", "1", "noloss", "502.29", "0.8"),
    ("PEPTIDEK", "PEPTIDEK", "2", "464.75", "25.5", "0.95", "PG1", "ProtA", "P001", "GENE1", "", "y", "4", "1", "H2O", "486.28", "0.2"),
    ("PEPTIDEK", "PEPTIDEK", "2", "464.75", "25.5", "0.95", "PG1", "ProtA", "P001", "GENE1", "", "b", "3", "1", "", "324.15", "0.5"),
    ("PEPTIDEK", "PEPTIDEK", "2", "464.75", "25.5", "0.95", "PG1", "ProtA", "P001", "GENE1", "", "b", "2", "1", "noloss", "227.10", "0.3"),
    ("PEPTIDEK", "PEPTIDEK", "2", "464.75", "25.5", "0.95", "PG1", "ProtA", "P001", "GENE1", "", "y", "2", "1", "noloss", "227.10", "0.4"),
    ("PEPTIDEK", "PEPTIDEK", "2", "464.75", "25.5", "0.95", "PG1", "ProtA", "P001", "GENE1", "", "y", "4", "1", "unknown", "502.30", "0.9"),
    # Decoys in every accepted spelling: "1", "1.0", "True".
    ("DECOYAPEK", "DECOYAPEK", "2", "500.11", "10.0", "0.9", "PGD", "ProtD", "PD1", "GENED", "1", "y", "3", "1", "", "350.10", "0.5"),
    ("DECOYBPEK", "DECOYBPEK", "2", "501.12", "11.0", "0.9", "PGD", "ProtD", "PD1", "GENED", "1.0", "y", "3", "1", "", "351.10", "0.5"),
    ("DECOYCPEK", "DECOYCPEK", "3", "502.13", "12.0", "0.9", "PGD", "ProtD", "PD1", "GENED", "True", "y", "3", "1", "", "352.10", "0.5"),
    # P3: contains X -> skipped.
    ("PEPXTIDEK", "PEPXTIDEK", "2", "510.20", "30.0", "0.9", "PG3", "Prot3", "P003", "GENE3", "0", "y", "4", "1", "", "503.29", "0.7"),
    # P4: contains Z (no defined mass) -> skipped.
    ("PEPZTIDEK", "PEPZTIDEK", "2", "511.21", "31.0", "0.9", "PG4", "Prot4", "P004", "GENE4", "0", "y", "4", "1", "", "504.29", "0.7"),
    # P5: DIA-NN N-terminal tag that must move behind the first residue, empty
    # Genes (-> '""'), IonMobility "0.0" (-> NaN), fragment charge 2.
    ("(tag)C(UniMod:4)SQAPVYGR", "CSQAPVYGR", "2", "540.26", "18.2", "0.0", "PG5", "Prot5", "P005", "", "0", "y", "5", "2", "", "289.66", "1.0"),
    # P6: underscore-wrapped ModifiedPeptide, IonMobility "" (-> NaN).
    ("_LIONELK_", "LIONELK", "1", "830.51", "40.1", "", "PG6", "Prot6", "P006", "GENE6", "", "b", "2", "1", "NH3", "210.09", "0.6"),
    # P7: IonMobility "0" (integer spelling) -> NaN, same as "0.0".
    ("SEVENPEPK", "SEVENPEPK", "2", "520.77", "33.3", "0", "PG7", "Prot7", "P007", "GENE7", "", "y", "3", "1", "", "375.20", "0.4"),
]

DIANN_COLUMNS = [
    "Modified.Sequence", "Stripped.Sequence", "Precursor.Charge",
    "Precursor.Mz", "Tr_recalibrated", "Protein.Group", "Protein.Names",
    "Protein.Ids", "Genes", "Fragment.Type", "Fragment.Series.Number",
    "Fragment.Charge", "Fragment.Loss.Type", "Product.Mz",
    "Relative.Intensity",
]

DIANN_ROWS = [
    ("ALIASPEPK", "ALIASPEPK", "2", "478.77", "22.0", "PGA", "ProtAlias", "PA01", "GENEA", "y", "4", "1", "noloss", "470.27", "0.9"),
    ("ALIASPEPK", "ALIASPEPK", "2", "478.77", "22.0", "PGA", "ProtAlias", "PA01", "GENEA", "b", "3", "1", "noloss", "282.18", "0.4"),
]

# ProteinID present but ProteinName absent: the single source column must
# populate both protein_name and uniprot_id.
PID_ONLY_COLUMNS = [
    "ModifiedPeptide", "StrippedPeptide", "PrecursorCharge", "PrecursorMz",
    "Tr_recalibrated", "ProteinID", "Genes", "FragmentType", "FragmentNumber",
    "FragmentCharge", "FragmentLossType", "FragmentMz", "RelativeIntensity",
]

PID_ONLY_ROWS = [
    ("ONLYIDPEK", "ONLYIDPEK", "2", "530.28", "15.0", "P00X", "GENEX", "y", "4", "1", "", "480.27", "0.8"),
]

# No retention-time column at all: parsing must raise.
MISSING_RT_COLUMNS = [
    "ModifiedPeptide", "StrippedPeptide", "PrecursorCharge", "PrecursorMz",
    "ProteinName", "FragmentType", "FragmentNumber", "FragmentCharge",
    "FragmentLossType", "FragmentMz", "RelativeIntensity",
]

MISSING_RT_ROWS = [
    ("NORTPEPK", "NORTPEPK", "2", "450.24", "ProtN", "y", "3", "1", "", "360.20", "0.5"),
]

# Columns that stay strings in the parquet variants (see module docstring).
STRING_ONLY = {"Decoy", "IonMobility"}
FLOAT_COLUMNS = {
    "PrecursorMz", "Precursor.Mz", "Tr_recalibrated", "FragmentMz",
    "Product.Mz", "RelativeIntensity", "Relative.Intensity",
}
INT_COLUMNS = {
    "PrecursorCharge", "Precursor.Charge", "FragmentNumber",
    "Fragment.Series.Number", "FragmentCharge", "Fragment.Charge",
}


def write_tsv(name, columns, rows):
    with open(os.path.join(HERE, name), "w", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(columns)
        writer.writerows(rows)


def write_parquet(name, columns, rows):
    arrays = []
    for i, col in enumerate(columns):
        values = [row[i] for row in rows]
        if col in STRING_ONLY:
            arrays.append(pa.array(values, type=pa.string()))
        elif col in FLOAT_COLUMNS:
            arrays.append(pa.array([float(v) for v in values], type=pa.float64()))
        elif col in INT_COLUMNS:
            arrays.append(pa.array([int(v) for v in values], type=pa.int64()))
        else:
            arrays.append(pa.array(values, type=pa.string()))
    pq.write_table(pa.table(arrays, names=columns), os.path.join(HERE, name))


def main():
    write_tsv("library_edgecases.tsv", EDGECASE_COLUMNS, EDGECASE_ROWS)
    write_parquet("library_edgecases.parquet", EDGECASE_COLUMNS, EDGECASE_ROWS)
    write_tsv("library_diann_aliases.tsv", DIANN_COLUMNS, DIANN_ROWS)
    write_parquet("library_diann_aliases.parquet", DIANN_COLUMNS, DIANN_ROWS)
    write_tsv("library_proteinid_only.tsv", PID_ONLY_COLUMNS, PID_ONLY_ROWS)
    write_parquet("library_proteinid_only.parquet", PID_ONLY_COLUMNS, PID_ONLY_ROWS)
    write_tsv("library_missing_rt.tsv", MISSING_RT_COLUMNS, MISSING_RT_ROWS)
    print(f"fixtures written to {HERE}")


if __name__ == "__main__":
    main()
