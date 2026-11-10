"""a-ion fragments in decoy generation (library_store._decoy_fragment_mz).

2.1.0 refused any library with a-ions ("Decoy generation only supports b/y fragment ions").
Decoy a_k is now the decoy's own b_k minus CO, from the same prefix sums.  Uses the first 300
precursors of the bundled data/filtered_library.tsv, with and without added a2 rows (a2 = b2 - CO):
  1. decoy a2 m/z == decoy b2 m/z - CO/z for every decoy that has both
  2. decoy b/y m/z are identical with and without the a2 rows (b/y path unchanged)
Run: pytest tests/test_aions_decoy.py   or   python tests/test_aions_decoy.py"""
import os, sys, tempfile
import polars as pl

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src.models.spec_lib import spec_lib
from src.utils.frag_encoding import get_ion_type, get_index, get_charge

LIB = os.path.join(ROOT, 'data', 'filtered_library.tsv')
CO = 27.99491461956


def _libraries(tmp, n=300):
    d = pl.read_csv(LIB, separator='\t', infer_schema_length=0)
    keep = d['transition_group_id'].unique(maintain_order=True)[:n]
    d = d.filter(pl.col('transition_group_id').is_in(keep.implode()))
    b2 = d.filter((pl.col('FragmentType') == 'b') & (pl.col('FragmentSeriesNumber') == '2') & (pl.col('FragmentCharge') == '1'))
    a2 = b2.with_columns(pl.lit('a').alias('FragmentType'),
                         (pl.col('FragmentMz').cast(pl.Float64) - CO).cast(pl.String).alias('FragmentMz'),
                         (pl.col('RelativeIntensity').cast(pl.Float64) * 0.124).cast(pl.String).alias('RelativeIntensity'))
    plain, with_a = os.path.join(tmp, 'plain.tsv'), os.path.join(tmp, 'with_a2.tsv')
    d.write_csv(plain, separator='\t')
    pl.concat([d, a2]).write_csv(with_a, separator='\t')
    return plain, with_a


def _decoy_frags(path):
    lib = spec_lib.loadSpecLib(path)[0]
    lib = spec_lib.create_decoy_lib(lib, rules='rev', tag=None)
    out = {}
    for e in range(lib.n_targets, lib.n_targets + lib.n_decoys):
        s, L = int(lib.frag_offsets[e]), int(lib.frag_lengths[e])
        c = lib.frag_keys_data[s:s + L]
        out[(lib.mod_seq[e], float(lib.prec_z[e]))] = {
            (int(t), int(i), int(z)): float(m)
            for t, i, z, m in zip(get_ion_type(c), get_index(c), get_charge(c), lib.frag_mz[s:s + L])}
    return out


def test_aion_decoys():
    with tempfile.TemporaryDirectory() as tmp:
        plain_f, with_a_f = _libraries(tmp)
        plain, with_a = _decoy_frags(plain_f), _decoy_frags(with_a_f)
    assert plain.keys() == with_a.keys() and len(plain) > 200, (len(plain), len(with_a))
    n_a = n_by = 0
    for k, fr in with_a.items():
        for (t, i, z), m in fr.items():
            if t == 2:
                b = fr[(0, i, z)]
                assert abs((b - m) - CO / z) < 2e-4, (k, i, z, b, m)
                n_a += 1
            else:
                assert plain[k][(t, i, z)] == m, (k, t, i, z, plain[k][(t, i, z)], m)
                n_by += 1
    assert n_a > 200
    print(f'{len(with_a)} decoys: {n_a} decoy a2 ions = decoy b2 - CO; '
          f'{n_by} decoy b/y ions identical with and without a2 rows')


if __name__ == '__main__':
    test_aion_decoys()
    print('a-ion decoys OK')
