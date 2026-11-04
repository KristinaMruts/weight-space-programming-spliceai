#!/usr/bin/env bash
# Re-run a participant's edit.py on the original weights and check that it reproduces the submitted
# model_edited.pt bit for bit (edit.py is kept exactly as submitted; it expects model/ and submission/ dirs).
#   bash scripts/reproduce_edit.sh opus
set -euo pipefail
cd "$(dirname "$0")/.."
who=$1
tmp=$(mktemp -d "${TMPDIR:-/tmp}/wsp.XXXXXX")
mkdir -p "$tmp/model" "$tmp/submission" "$tmp/code"
cp weights/model_10000nt_rs10.pt "$tmp/model/"
cp "submissions/$who/edit.py" "$tmp/code/edit.py"
cp scripts/osai.py scripts/evaluate.py "$tmp/code/" 2>/dev/null || true
cp -r data "$tmp/data"
(cd "$tmp/code" && python edit.py)
python - "$tmp/submission/model_edited.pt" "submissions/$who/model_edited.pt" <<'EOF'
import sys, torch
a = torch.load(sys.argv[1], map_location="cpu", weights_only=True)
b = torch.load(sys.argv[2], map_location="cpu", weights_only=True)
assert a.keys() == b.keys()
same = all(torch.equal(a[k], b[k]) for k in a)
print("bit-identical to the submitted weights:", same)
sys.exit(0 if same else 1)
EOF
rm -rf "$tmp"
