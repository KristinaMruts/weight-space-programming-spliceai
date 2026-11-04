"""Build the benchmark data: OpenSpliceAI x ClinVar near-splice variants.

Variants: ClinVar SNVs (GRCh38, release 2026-11-01), pathogenic/likely pathogenic vs benign/likely benign,
review status >= 1 star, molecular consequence intronic / synonymous / splice-region (no missense, nonsense,
etc.), within 50 nt of an internal exon boundary of a MANE Select transcript, canonical dinucleotides
excluded (donor +1/+2, acceptor -1/-2). Windows of 10,101 nt in transcript orientation, N outside the
transcript (as in OpenSpliceAI variant scoring). At most two benign per pathogenic, sampled with fixed seeds.

Split by chromosome: dev = chr17, chr19, chr21, chr22 (given to the participants);
test = chr1, chr11, chr16 (hidden during the runs).

Guard: whole MANE genes, task = find ordinary splice sites (top-k accuracy as in the SpliceAI paper).
dev guard = 60 genes from dev chromosomes; test guard = 150 genes from chr1 (held out from training in the
SpliceAI / OpenSpliceAI split).

  python build_data.py --raw raw --out data
"""
from __future__ import annotations

import argparse
import gzip
import io
import json
import random
import re
from collections import defaultdict
from pathlib import Path

DEV = ["chr17", "chr19", "chr21", "chr22"]
TEST = ["chr1", "chr11", "chr16"]
FLANK = 5000
HALF = 5050
RC = str.maketrans("ACGTN", "TGCAN")
P_SIG = {"Pathogenic", "Likely_pathogenic", "Pathogenic/Likely_pathogenic"}
B_SIG = {"Benign", "Likely_benign", "Benign/Likely_benign"}
OK_REV = {"criteria_provided,_single_submitter", "criteria_provided,_multiple_submitters,_no_conflicts",
          "reviewed_by_expert_panel", "practice_guideline"}
KEEP_MC = {"intron_variant", "synonymous_variant", "splice_region_variant", "splice_donor_5th_base_variant",
           "splice_donor_region_variant", "splice_polypyrimidine_tract_variant"}
BAD_MC = {"missense_variant", "nonsense", "stop_gained", "stop_lost", "start_lost", "frameshift_variant",
          "splice_donor_variant", "splice_acceptor_variant", "initiator_codon_variant", "inframe_deletion",
          "inframe_insertion", "5_prime_UTR_variant", "3_prime_UTR_variant", "non-coding_transcript_variant"}


def read_fa(path):
    with gzip.open(path, "rt") as f:
        next(f)
        return "".join(l.strip() for l in f).upper()


def mane(raw):
    tx, exons = {}, defaultdict(list)
    rx = re.compile(r'(\S+) "?([^";]*)"?;')
    with gzip.open(raw / "gencode.gtf.gz", "rt") as f:
        for line in f:
            if line[0] == "#":
                continue
            c = line.split("\t", 8)
            if c[0] not in DEV + TEST or c[2] not in ("transcript", "exon") or "MANE_Select" not in c[8]:
                continue
            a = defaultdict(list)
            for k, v in rx.findall(c[8]):
                a[k].append(v)
            tid = a["transcript_id"][0]
            if c[2] == "transcript":
                tx[tid] = dict(chrom=c[0], start=int(c[3]), end=int(c[4]), strand=c[6], gene=a["gene_name"][0])
            else:
                exons[tid].append((int(c[3]), int(c[4])))
    for tid in tx:
        tx[tid]["exons"] = sorted(exons[tid])
    return tx


def boundaries(t):
    """Internal boundaries in genome coordinates: (D, last exon base) and (A, first exon base)."""
    ex = t["exons"]
    out = []
    for (s1, e1), (s2, e2) in zip(ex, ex[1:]):
        if t["strand"] == "+":
            out.append(("D", e1)); out.append(("A", s2))
        else:  # minus strand: the donor is the start of the exon to the right
            out.append(("A", e1)); out.append(("D", s2))
    return out


def offset(t, pos):
    """Nearest internal boundary and offset in transcript orientation.
    Donor: exon ... -2 -1 | +1 +2 ... intron. Acceptor: intron ... -2 -1 | +1 +2 ... exon."""
    best = None
    sgn = 1 if t["strand"] == "+" else -1
    for kind, b in boundaries(t):
        d = (pos - b) * sgn          # > 0: downstream in the transcript
        if kind == "D":
            off = d if d > 0 else d - 1   # b = last exon base -> -1
        else:
            off = d + 1 if d >= 0 else d  # b = first exon base -> +1
        if best is None or abs(off) < abs(best[1]):
            best = (kind, off)
    return best


def window(g, t, pos):
    s = g[pos - 1 - HALF: pos + HALF]
    lo, hi = t["start"], t["end"]
    s = "".join(ch if lo <= pos - HALF + i <= hi else "N" for i, ch in enumerate(s))
    return s if t["strand"] == "+" else s.translate(RC)[::-1]


def parse_clinvar(raw, chroms):
    out = []
    with gzip.open(raw / "clinvar.vcf.gz", "rt") as f:
        for line in f:
            if line[0] == "#":
                continue
            c = line.rstrip("\n").split("\t")
            chrom = "chr" + c[0]
            if chrom not in chroms or len(c[3]) != 1 or len(c[4]) != 1 or c[4] not in "ACGT":
                continue
            info = dict(kv.split("=", 1) for kv in c[7].split(";") if "=" in kv)
            sig = info.get("CLNSIG", "")
            lab = 1 if sig in P_SIG else 0 if sig in B_SIG else None
            if lab is None or info.get("CLNREVSTAT") not in OK_REV:
                continue
            mc = {x.split("|")[1] for x in info.get("MC", "").split(",") if "|" in x}
            if not mc & KEEP_MC or mc & BAD_MC:
                continue
            stars = {"criteria_provided,_single_submitter": 1, "criteria_provided,_multiple_submitters,_no_conflicts": 2,
                     "reviewed_by_expert_panel": 3, "practice_guideline": 4}[info["CLNREVSTAT"]]
            out.append(dict(id=c[2], chrom=chrom, pos=int(c[1]), ref=c[3], alt=c[4], label=lab,
                            mc=",".join(sorted(mc)), stars=stars, gene_cv=info.get("GENEINFO", "").split(":")[0]))
    return out


def variants(raw, tx, chroms, rng):
    by_chrom = defaultdict(list)
    for tid, t in tx.items():
        by_chrom[t["chrom"]].append(t)
    rows = []
    cv = parse_clinvar(raw, set(chroms))
    for chrom in chroms:
        g = read_fa(raw / f"{chrom}.fa.gz")
        ts = by_chrom[chrom]
        for v in (x for x in cv if x["chrom"] == chrom):
            hits = [t for t in ts if t["start"] <= v["pos"] <= t["end"]]
            if not hits:
                continue
            t = hits[0]
            bo = offset(t, v["pos"])
            if bo is None:  # single-exon transcript: no internal boundaries
                continue
            kind, off = bo
            if abs(off) > 50 or (kind == "D" and off in (1, 2)) or (kind == "A" and off in (-1, -2)):
                continue
            if g[v["pos"] - 1] != v["ref"]:
                continue
            w = window(g, t, v["pos"])
            alt = v["alt"] if t["strand"] == "+" else v["alt"].translate(RC)
            ref = v["ref"] if t["strand"] == "+" else v["ref"].translate(RC)
            assert w[HALF] == ref
            rows.append(dict(v, strand=t["strand"], gene=t["gene"], site=f"{kind}{off:+d}",
                             ref_tx=ref, alt_tx=alt, window=w))
        del g
    P = [r for r in rows if r["label"] == 1]
    B = [r for r in rows if r["label"] == 0]
    rng.shuffle(B)
    B = B[:2 * len(P)]
    out = P + B
    rng.shuffle(out)
    print(chroms, "P", len(P), "B kept", len(B), "of", len([r for r in rows if r['label'] == 0]))
    return out


def guard(raw, tx, chroms, n, rng):
    ts = [t for t in tx.values() if t["chrom"] in chroms and len(t["exons"]) >= 3 and t["end"] - t["start"] < 100_000]
    rng.shuffle(ts)
    ts = ts[:n]
    out = []
    for chrom in chroms:
        sel = [t for t in ts if t["chrom"] == chrom]
        if not sel:
            continue
        g = read_fa(raw / f"{chrom}.fa.gz")
        for t in sel:
            s = g[t["start"] - 1 - FLANK: t["end"] + FLANK]
            L = t["end"] - t["start"] + 1
            acc, don = [], []
            for kind, b in boundaries(t):
                i = b - t["start"] if t["strand"] == "+" else t["end"] - b
                (acc if kind == "A" else don).append(i)
            if t["strand"] == "-":
                s = s.translate(RC)[::-1]
            out.append(dict(gene=t["gene"], chrom=chrom, strand=t["strand"], length=L,
                            acceptors=sorted(acc), donors=sorted(don), seq=s))
        del g
    print(chroms, "guard genes", len(out))
    return out


COLS = ["id", "chrom", "pos", "ref", "alt", "strand", "gene", "site", "mc", "stars", "label", "ref_tx", "alt_tx", "window"]


def _gz(path):  # mtime=0 so that rebuilt files are byte-identical
    return io.TextIOWrapper(gzip.GzipFile(path, "wb", mtime=0), encoding="utf-8")


def write_tsv(rows, path):
    with _gz(path) as f:
        f.write("\t".join(COLS) + "\n")
        for r in rows:
            f.write("\t".join(str(r[c]) for c in COLS) + "\n")


def write_jsonl(rows, path):
    with _gz(path) as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="raw"); ap.add_argument("--out", default="data")
    a = ap.parse_args()
    raw, out = Path(a.raw), Path(a.out)
    (out / "dev").mkdir(parents=True, exist_ok=True); (out / "test").mkdir(parents=True, exist_ok=True)
    tx = mane(raw)
    print("MANE transcripts", len(tx))
    dev = variants(raw, tx, DEV, random.Random(1))
    write_tsv(dev, out / "dev/dev_variants.tsv.gz")
    # quick subset for iterations: first 200 P + 400 B
    P = [r for r in dev if r["label"] == 1][:200]; B = [r for r in dev if r["label"] == 0][:400]
    write_tsv(P + B, out / "dev/dev_fast_variants.tsv.gz")
    write_jsonl(guard(raw, tx, DEV, 60, random.Random(2)), out / "dev/dev_guard_genes.jsonl.gz")
    test = variants(raw, tx, TEST, random.Random(3))
    write_tsv(test, out / "test/test_variants.tsv.gz")
    write_jsonl(guard(raw, tx, ["chr1"], 150, random.Random(4)), out / "test/test_guard_genes.jsonl.gz")


if __name__ == "__main__":
    main()
