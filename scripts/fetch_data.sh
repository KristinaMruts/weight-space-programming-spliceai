#!/usr/bin/env bash
# Download everything needed to rebuild the benchmark and to score models, and verify checksums.
#   raw/      GRCh38 chromosomes (UCSC), GENCODE v46 basic annotation, ClinVar GRCh38 release 2026-11-01
#   weights/  OpenSpliceAI-MANE-10000nt, seeds 10..14 (GPL-3.0, github.com/Kuanhao-Chao/OpenSpliceAI)
# Total ~0.5 GB. The built data is already in data/, so raw/ is only needed to check that it rebuilds.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p raw weights

get() {  # resumable download with retries
  local url=$1 out=$2
  for i in 1 2 3 4 5; do
    curl -fsSL -C - -o "$out" "$url" && return 0
    echo "retry $i: $url" >&2; sleep 3
  done
  return 1
}

OSAI=https://raw.githubusercontent.com/Kuanhao-Chao/OpenSpliceAI/main/models/openspliceai-mane/10000nt
for s in 10 11 12 13 14; do
  [ -s weights/model_10000nt_rs$s.pt ] || get $OSAI/model_10000nt_rs$s.pt weights/model_10000nt_rs$s.pt
done

if [ "${1:-}" = "--weights-only" ]; then
  (cd weights && shasum -a 256 -c ../checksums/weights.sha256)
  exit 0
fi

for c in chr1 chr11 chr16 chr17 chr19 chr21 chr22; do
  [ -s raw/$c.fa.gz ] || get https://hgdownload.soe.ucsc.edu/goldenPath/hg38/chromosomes/$c.fa.gz raw/$c.fa.gz
done
[ -s raw/gencode.gtf.gz ] || get https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_46/gencode.v46.basic.annotation.gtf.gz raw/gencode.gtf.gz
# Pinned ClinVar release. NCBI keeps dated weekly files for a while, then moves them to archive_2.0/<year>/.
CV=clinvar_20261101.vcf.gz
[ -s raw/clinvar.vcf.gz ] || get https://ftp.ncbi.nlm.nih.gov/pub/clinvar/vcf_GRCh38/$CV raw/clinvar.vcf.gz \
  || get https://ftp.ncbi.nlm.nih.gov/pub/clinvar/vcf_GRCh38/weekly/$CV raw/clinvar.vcf.gz \
  || get https://ftp.ncbi.nlm.nih.gov/pub/clinvar/vcf_GRCh38/archive_2.0/2026/$CV raw/clinvar.vcf.gz

(cd weights && shasum -a 256 -c ../checksums/weights.sha256)
(cd raw && shasum -a 256 -c ../checksums/raw.sha256)
