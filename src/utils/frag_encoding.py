"""
Packed int32 encoding for fragment ion names.

Fragment name format: (ion_type)(index)(-loss)_(charge)(_isoN)
Examples: b5_2, y10-H2O_1, b3_iso1, y7-NH3_2_iso3

Bit layout (21 of 32 bits used, 11 spare):

  Field       Bits    Shift   Mask    Capacity    Used    Spare
  ──────────  ──────  ──────  ──────  ──────────  ──────  ──────
  ion_type    3       0       0x7     8 values    6       2
  index       8       3       0xFF    0-255       ~1-50   ~200
  loss        3       11      0x7     8 values    4       4
  charge      3       14      0x7     1-7*        ~1-4    3
  iso         4       17      0xF     0-15        ~0-5    10
  SPARE       11      21      0x7FF   2048        0       2048
  ──────────  ──────
  TOTAL       32 bits                             21 used, 11 spare

  * charge stored as charge-1, so raw 0-6 maps to charge 1-7

Ion type table:  b=0, y=1, a=2, c=3, x=4, z=5  (2 spare)
Loss table:      none=0, H2O=1, NH3=2, H3PO4=3  (4 spare)
"""
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

import re
import numpy as np

# ---------------------------------------------------------------------------
# Ion type encoding: 3 bits [0:2], 6 used, 2 spare
# ---------------------------------------------------------------------------
_ION_TYPE_TO_INT = {'b': 0, 'y': 1, 'a': 2, 'c': 3, 'x': 4, 'z': 5}
_INT_TO_ION_TYPE = {v: k for k, v in _ION_TYPE_TO_INT.items()}

# ---------------------------------------------------------------------------
# Neutral loss encoding: 3 bits [11:13], 4 used, 4 spare
# ---------------------------------------------------------------------------
_LOSS_TO_INT = {'': 0, 'H2O': 1, 'NH3': 2, 'H3PO4': 3}
_INT_TO_LOSS = {v: k for k, v in _LOSS_TO_INT.items()}

# ---------------------------------------------------------------------------
# Bit layout — 21 of 32 bits used, 11 spare
#
#   [31:21]  11 bits  SPARE (unused, reserved for future fields)
#   [20:17]   4 bits  iso       0-15   isotope index (0 = monoisotopic)
#   [16:14]   3 bits  charge    0-6    stored as charge-1 (real charge 1-7)
#   [13:11]   3 bits  loss      0-7    neutral loss type (see table above)
#   [10:3]    8 bits  index     0-255  fragment ordinal
#   [2:0]     3 bits  ion_type  0-7    ion series (see table above)
# ---------------------------------------------------------------------------
_ION_SHIFT = 0       # bits [2:0]    — 3 bits, capacity 8
_IDX_SHIFT = 3       # bits [10:3]   — 8 bits, capacity 256
_LOSS_SHIFT = 11     # bits [13:11]  — 3 bits, capacity 8
_CHG_SHIFT = 14      # bits [16:14]  — 3 bits, capacity 8  (charge-1)
_ISO_SHIFT = 17      # bits [20:17]  — 4 bits, capacity 16
# bits [31:21] — 11 spare bits

_ION_MASK = 0x7       # 3 bits  → 0b111
_IDX_MASK = 0xFF      # 8 bits  → 0b1111_1111
_LOSS_MASK = 0x7      # 3 bits  → 0b111
_CHG_MASK = 0x7       # 3 bits  → 0b111
_ISO_MASK = 0xF       # 4 bits  → 0b1111

# Pre-compiled regex for parsing fragment name strings
_FRAG_RE = re.compile(r'^([abcxyz])(\d+)(?:-([^_]+))?_(\d+)(?:_iso(\d+))?$')

# Named constants for ion types (for fast vectorized comparisons)
ION_B = np.int32(0)
ION_Y = np.int32(1)


# ---- Single-value encode/decode ----

def encode_frag_name(name):
    """Encode a single fragment name string to int32.

    Parameters
    ----------
    name : str
        Fragment name, e.g. "b5_2", "y10-H2O_1_iso3"

    Returns
    -------
    np.int32
        Packed int32 code.
    """
    m = _FRAG_RE.match(name)
    if m is None:
        raise ValueError(f"Cannot parse fragment name: {name!r}")

    ion_type = _ION_TYPE_TO_INT[m.group(1)]
    index = int(m.group(2))
    loss = _LOSS_TO_INT.get(m.group(3) or '', 0)
    charge = int(m.group(4)) - 1  # store 1-7 as 0-6
    iso = int(m.group(5)) if m.group(5) else 0

    return np.int32(
        (ion_type << _ION_SHIFT)
        | (index << _IDX_SHIFT)
        | (loss << _LOSS_SHIFT)
        | (charge << _CHG_SHIFT)
        | (iso << _ISO_SHIFT)
    )


_decode_cache = {}

def decode_frag_name(code):
    """Decode a packed int32 code back to a fragment name string.

    Parameters
    ----------
    code : int or np.int32
        Packed fragment code.

    Returns
    -------
    str
        Fragment name, e.g. "b5_2", "y10-H2O_1_iso3"
    """
    code = int(code)
    cached = _decode_cache.get(code)
    if cached is not None:
        return cached
    ion_type = _INT_TO_ION_TYPE[(code >> _ION_SHIFT) & _ION_MASK]
    index = (code >> _IDX_SHIFT) & _IDX_MASK
    loss = _INT_TO_LOSS.get((code >> _LOSS_SHIFT) & _LOSS_MASK, '')
    charge = ((code >> _CHG_SHIFT) & _CHG_MASK) + 1
    iso = (code >> _ISO_SHIFT) & _ISO_MASK

    name = ion_type + str(index)
    if loss:
        name += '-' + loss
    name += '_' + str(charge)
    if iso > 0:
        name += '_iso' + str(iso)
    _decode_cache[code] = name
    return name


# ---- Bulk encode/decode ----

def encode_frag_names(names):
    """Encode an array/list of fragment name strings to int32 numpy array.

    Parameters
    ----------
    names : array-like of str
        Fragment name strings.

    Returns
    -------
    np.ndarray[int32]
        Packed codes.
    """
    n = len(names)
    if n == 0:
        return np.empty(0, dtype=np.int32)
    codes = np.empty(n, dtype=np.int32)
    for i in range(n):
        codes[i] = encode_frag_name(names[i])
    return codes


def decode_frag_names(codes):
    """Decode an int32 array of packed codes to object array of strings.

    Parameters
    ----------
    codes : np.ndarray[int32]
        Packed fragment codes.

    Returns
    -------
    np.ndarray[object]
        Fragment name strings.
    """
    n = len(codes)
    if n == 0:
        return np.empty(0, dtype=object)
    out = np.empty(n, dtype=object)
    for i in range(n):
        out[i] = decode_frag_name(codes[i])
    return out


# ---- Vectorized bitfield queries (operate on int32 arrays) ----

def is_b_ion(codes):
    """Boolean mask: True where ion type is 'b'."""
    return ((np.asarray(codes) >> _ION_SHIFT) & _ION_MASK) == ION_B


def is_y_ion(codes):
    """Boolean mask: True where ion type is 'y'."""
    return ((np.asarray(codes) >> _ION_SHIFT) & _ION_MASK) == ION_Y


def is_isotope(codes):
    """Boolean mask: True where iso > 0."""
    return ((np.asarray(codes) >> _ISO_SHIFT) & _ISO_MASK) != 0


def get_index(codes):
    """Array of fragment indices (ordinals)."""
    return (np.asarray(codes) >> _IDX_SHIFT) & _IDX_MASK


def get_ion_type(codes):
    """Array of ion type codes (0=b, 1=y, 2=a, ...)."""
    return (np.asarray(codes) >> _ION_SHIFT) & _ION_MASK


def get_charge(codes):
    """Array of charge values (1-based)."""
    return ((np.asarray(codes) >> _CHG_SHIFT) & _CHG_MASK) + 1


def get_loss(codes):
    """Array of loss type codes (0=none, 1=H2O, 2=NH3, 3=H3PO4)."""
    return (np.asarray(codes) >> _LOSS_SHIFT) & _LOSS_MASK


def get_iso(codes):
    """Array of isotope indices (0=monoisotopic)."""
    return (np.asarray(codes) >> _ISO_SHIFT) & _ISO_MASK
