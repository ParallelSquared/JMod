#!/usr/bin/env python
"""Replace JMod's copy of Unimod (data/unimod.xml) with the latest from unimod.org.

JMod reads modifications from that copy, so a search needs no network.  This
downloads the current file, checks pyteomics can read it, and replaces the
copy only then, reporting how many modifications were added or removed.
Commit the new data/unimod.xml to ship it.

Usage::

    python scripts/update_unimod.py
"""

import os
import shutil
import sys
import tempfile
import urllib.request

from pyteomics import mass

UNIMOD_URL = "https://www.unimod.org/xml/unimod.xml"
UNIMOD_XML = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "unimod.xml")


def record_ids(path):
    return {mod["record_id"] for mod in mass.Unimod(path).mods}


def main():
    with tempfile.TemporaryDirectory() as tmp:
        downloaded = os.path.join(tmp, "unimod.xml")
        with urllib.request.urlopen(UNIMOD_URL, timeout=60) as response, open(downloaded, "wb") as f:
            f.write(response.read())
        new = record_ids(downloaded)  # raises if pyteomics cannot read it, leaving the copy as it was
        old = record_ids(UNIMOD_XML) if os.path.exists(UNIMOD_XML) else set()
        shutil.move(downloaded, UNIMOD_XML)  # the temp folder may be on another drive
    print(f"{UNIMOD_XML}: {len(new)} modifications "
          f"({len(new - old)} added, {len(old - new)} removed since the previous copy)")


if __name__ == "__main__":
    sys.exit(main())
