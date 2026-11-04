"""Check that data/ (or a rebuilt copy) matches the published benchmark: SHA-256 of decompressed contents.

  python scripts/check_data.py            # data/ as shipped
  python scripts/check_data.py rebuilt    # after: python scripts/build_data.py --raw raw --out rebuilt
"""
import gzip
import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
base = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "data"
ok = True
for line in (ROOT / "checksums/data_content.sha256").read_text().split("\n"):
    if not line.strip():
        continue
    want, rel = line.split()
    h = hashlib.sha256()
    with gzip.open(base / rel, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    good = h.hexdigest() == want
    ok &= good
    print(("OK  " if good else "BAD ") + rel)
sys.exit(0 if ok else 1)
