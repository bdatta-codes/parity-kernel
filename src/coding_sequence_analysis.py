#!/usr/bin/env python3
"""
coding_sequence_analysis.py
===========================

The annotated-coding-sequence analyses of the paper

    "A Closed-Form Expectation for Intra-Strand Parity Indices and the Limits of Their Use as G-Quadruplex Descriptors"  (Arya, Pancholi & Datta)

The prediction tested (Introduction) is that, because codons are mostly NOT
their own reverses, a parity
index should detect reading frame strongly and, in the correct frame, depend
only on the codon count table. This script reproduces every coding-sequence
number in the Results:

  (1) FIGURE 5A  -- separation of real coding sequence from length-matched real
      non-coding sequence (introns + UTR flanks of the SAME loci) as a function
      of ORF length, reported as ROC-AUC in ORF-length bins, for
        - the FRAME-RESOLVED index (frame-0, non-overlapping codons),
        - the FRAME-FREE index (sliding-window trinucleotides),
        - Fickett's position-asymmetry statistic, and
        - a hexamer log-likelihood ratio scored out of fold
      (the last two from coding_comparators.py; see its header for what is and
      is not implemented, and run its --selftest for the validation cases).
      Each coding sequence is paired with a non-coding window of exactly its own
      length, and the draw of windows is repeated (default 200 times); every
      AUC is reported as the mean over draws with a 2.5-97.5% interval.
      Direction: coding sequence shows GREATER reverse-pair imbalance than
      non-coding sequence of the same length.

  (2) FIGURE 5B  -- the frame-0 index is a codon-usage statistic: on all
      1,076 coding sequences (20 shuffles each) the mean frame-0 index of the
      real sequence, after permuting whole codons (identical; max |delta| = 0),
      after shuffling bases within each codon position, and after a mononucleotide shuffle. The fraction of the
      depression (mononucleotide-shuffled minus real) RECOVERED by the
      within-position shuffle is the part due to codon identity beyond
      per-position composition; the remainder is carried by per-position
      composition, which Fickett's statistic reads. -> fig5b_codon_structure.csv

  (3) FRAME IDENTIFICATION -- which of the three frames gives the lowest
      frame value (true frame, shifted by one, shifted by two), by ORF-length bin.
      -> frame_identification_by_orf_length.csv, frame_values.csv

  (4) FICKETT vs SYMMETRY -- Spearman rho and five-fold cross-validated
      logistic AUCs (Fickett alone, symmetry features alone, both), on 660
      coding + 660 exact-length non-coding sequences, with sensitivity rows.
      -> fickett_vs_symmetry_logistic.csv

  (2)-(4) are implemented in coding_extras.py, whose header states every
  definition; each uses its own fixed seed. Figure 5 (both panels) is drawn
  by make_fig5.py from the fig5a and fig5b CSVs.

THE DATA USED IN THE PAPER
--------------------------
The annotated loci are the AUGUSTUS training GenBank files for Drosophila
melanogaster chromosomes 2L (492 loci) and 2R (586 loci), one gene per LOCUS
with its flanking sequence (Stanke & Waack 2003). They are not redistributed
here; data/fetch_augustus_fly.sh downloads them from a pinned AUGUSTUS commit
and checks their MD5 sums. Then:

    bash data/fetch_augustus_fly.sh
    python3 src/coding_sequence_analysis.py \\
        --genbank data/fly2L.gb data/fly2R.gb --out results/cds/ --bins default
    python3 src/coding_sequence_analysis.py \\
        --genbank data/fly2L.gb data/fly2R.gb --out results/cds/ --bins hexamer

This yields exactly the sets analysed: 1,076 coding sequences (40-5,522
codons, median 386.5) after removing two exact duplicates, and 2,952 non-coding
pieces. A genome FASTA plus GFF3 can be supplied instead (--fasta/--gff) to run
the same analyses on other annotations.

The negative (non-coding) set is built from the introns and the 5' and 3'
untranslated flanks of the SAME loci that supply the coding sequences, each at
least 120 nt (--min-noncoding), so that positives and negatives share local
composition, repeat structure and isochore context — a real-sequence control,
harder than a shuffle.

CDS acceptance filters (as in the paper): the terminal stop codon is removed
if annotated (both GenBank conventions occur), then a whole number of codons,
an ATG start, only ACGT characters, and no in-frame stop codon.

WHY THIS SCRIPT IS NOT IN run_all.sh
------------------------------------
It needs the downloaded loci above; the whole run, including the 200-draw
Figure 5A analysis for one binning, takes about 20 minutes on one core.

Dependencies: numpy, scipy, pandas, biopython (Bio.SeqIO for GenBank and
FASTA; a light GFF3 reader is included so gffutils is not required). The annotation must be
GFF3 (attributes of the form ID=...;Parent=...); GTF is not parsed.

Authors: Arjun Arya, Heli Pancholi, Bhaskar Datta
         Indian Institute of Technology Gandhinagar
License: MIT
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import parity  # noqa: E402
from coding_comparators import fickett_asymmetry, hexamer_cv_auc

_STOPS = {"TAA", "TAG", "TGA"}
_COMP = str.maketrans("ACGTacgt", "TGCAtgca")


def revcomp(s: str) -> str:
    return s.translate(_COMP)[::-1]


# --------------------------------------------------------------------------- #
# Frame-resolved (frame-0, non-overlapping codons) vs frame-free nMSI
# --------------------------------------------------------------------------- #
def nmsi_frame0(seq: str) -> float:
    """nMSI over NON-OVERLAPPING trinucleotides starting at position 0 — i.e.
    the codons of a coding sequence in its correct frame. N ~ L/3. This is the
    'frame-resolved' index of Figure 5."""
    s = seq.upper()
    codons = [s[i:i + 3] for i in range(0, len(s) - 2, 3)]
    codons = [c for c in codons if all(b in "ACGT" for b in c)]
    if not codons:
        return float("nan")
    freq: Dict[str, int] = {}
    for c in codons:
        freq[c] = freq.get(c, 0) + 1
    num = den = 0.0
    seen = set()
    for w in freq:
        rw = w[::-1]
        key = tuple(sorted([w, rw]))
        if key in seen:
            continue
        seen.add(key)
        fw, frw = freq.get(w, 0), freq.get(rw, 0)
        if w != rw:                     # non-palindromic only (nMSI)
            num += abs(fw - frw)
            den += fw + frw
    return 1.0 - num / den if den > 0 else float("nan")


def nmsi_framefree(seq: str) -> float:
    """The ordinary sliding-window nMSI (parity.msi_scores) — the 'frame-free'
    index of Figure 5."""
    _, nm = parity.msi_scores(seq)
    return nm if nm is not None else float("nan")


# --------------------------------------------------------------------------- #
# Codon-structure controls (Figure 5B)
# --------------------------------------------------------------------------- #
def permute_codons(seq: str, rng: np.random.Generator) -> str:
    """Shuffle whole codons; codon counts unchanged, codon order destroyed."""
    s = seq.upper()
    codons = [s[i:i + 3] for i in range(0, len(s) - len(s) % 3, 3)]
    rng.shuffle(codons)
    return "".join(codons)


def shuffle_within_codon_position(seq: str, rng: np.random.Generator) -> str:
    """Shuffle bases within each codon POSITION (all position-1 bases among
    themselves, etc.). Preserves the per-position base composition that
    Fickett's statistic reads, while destroying codon identity."""
    s = seq.upper()
    L = len(s) - len(s) % 3
    cols = [list(s[p:L:3]) for p in range(3)]
    for col in cols:
        rng.shuffle(col)
    out = []
    for i in range(L // 3):
        out.extend([cols[0][i], cols[1][i], cols[2][i]])
    return "".join(out)


# --------------------------------------------------------------------------- #
# Minimal GFF3 reader: gene -> CDS segments, introns, UTR flanks
# --------------------------------------------------------------------------- #
class Locus:
    __slots__ = ("gene_id", "seqid", "strand", "cds", "exons", "start", "end")

    def __init__(self, gene_id, seqid, strand):
        self.gene_id = gene_id
        self.seqid = seqid
        self.strand = strand
        self.cds: List[Tuple[int, int]] = []      # 0-based half-open
        self.exons: List[Tuple[int, int]] = []
        self.start = None
        self.end = None


def read_gff_loci(gff_path: str) -> Dict[str, Locus]:
    """Group CDS and exon features under their parent transcript/gene. Accepts
    GFF3 (attributes 'ID=', 'Parent=') well enough for standard annotations."""
    loci: Dict[str, Locus] = {}
    parent_to_gene: Dict[str, str] = {}

    def attr(field: str, key: str):
        for kv in field.replace(" ", "").split(";"):
            if kv.startswith(key + "="):
                return kv[len(key) + 1:]
        return None

    opener = open
    if gff_path.endswith(".gz"):
        import gzip
        opener = lambda p: gzip.open(p, "rt")
    with opener(gff_path) as fh:
        for line in fh:
            if line.startswith("#") or not line.strip():
                continue
            f = line.rstrip("\n").split("\t")
            if len(f) < 9:
                continue
            seqid, _src, ftype, start, end, _sc, strand, _fr, attrs = f[:9]
            start = int(start) - 1  # to 0-based
            end = int(end)          # half-open
            if ftype in ("mRNA", "transcript"):
                tid = attr(attrs, "ID")
                gid = attr(attrs, "Parent") or tid
                if tid:
                    parent_to_gene[tid] = gid or tid
            elif ftype in ("CDS", "exon"):
                parent = attr(attrs, "Parent") or attr(attrs, "ID") or "unk"
                gid = parent_to_gene.get(parent, parent)
                key = gid
                loc = loci.get(key)
                if loc is None:
                    loc = Locus(gid, seqid, strand)
                    loci[key] = loc
                (loc.cds if ftype == "CDS" else loc.exons).append((start, end))
                loc.start = start if loc.start is None else min(loc.start, start)
                loc.end = end if loc.end is None else max(loc.end, end)
    return {k: v for k, v in loci.items() if v.cds}


def _splice(seq: str, segments: List[Tuple[int, int]], strand: str) -> str:
    segs = sorted(segments)
    joined = "".join(seq[s:e] for s, e in segs)
    return revcomp(joined) if strand == "-" else joined.upper()


def cds_passes(cds_seq: str) -> bool:
    s = cds_seq.upper()
    if len(s) < 3 or len(s) % 3 != 0:
        return False
    if any(b not in "ACGT" for b in s):
        return False
    if s[:3] != "ATG":
        return False
    for i in range(0, len(s), 3):         # terminal stop already removed
        if s[i:i + 3] in _STOPS:
            return False
    return True


def _strip_terminal_stop(s: str) -> str:
    """Accept both GenBank conventions for the terminal stop codon: if the
    annotated CDS ends in a stop, remove it so the sequence is sense codons only."""
    return s[:-3] if len(s) >= 3 and s[-3:] in _STOPS else s


def _dedup(seqs: List[str]) -> List[str]:
    """Drop exact duplicate sequences, keeping first occurrence and order."""
    return list(dict.fromkeys(seqs))


def build_sets_genbank(gb_paths: List[str], min_noncoding: int = 120
                       ) -> Tuple[List[str], List[str]]:
    """Return (coding_sequences, noncoding_pieces) from GenBank training files
    of the AUGUSTUS kind, one gene per LOCUS with flanking sequence.

    Coding: the CDS feature spliced from its exon coordinates and placed on the
    coding strand, terminal stop removed, then required to be a whole number of
    codons, to begin ATG, to be ACGT only and to contain no stop codon; exact
    duplicates are removed.
    Non-coding: the introns and the 5' and 3' untranslated flanks of the same
    loci (sequence of the locus outside the CDS span), each at least
    `min_noncoding` nt and ACGT only; exact duplicates are removed.

    For the D. melanogaster chromosome 2L and 2R training files distributed with
    AUGUSTUS this yields 1,076 coding sequences (40-5,522 codons, median 386.5)
    and 2,952 non-coding pieces, the sets analysed in the paper."""
    import warnings
    try:
        from Bio import SeqIO
    except ImportError:
        sys.exit("[error] biopython is required (pip install biopython)")
    coding, noncoding = [], []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")      # AUGUSTUS LOCUS lines are terse
        for path in gb_paths:
            for rec in SeqIO.parse(path, "genbank"):
                seq = str(rec.seq).upper()
                for feat in (f for f in rec.features if f.type == "CDS"):
                    cds = _strip_terminal_stop(str(feat.extract(rec.seq)).upper())
                    if cds_passes(cds):
                        coding.append(cds)
                    parts = sorted((int(q.start), int(q.end)) for q in feat.location.parts)
                    pieces = [seq[b1:a2] for (_a1, b1), (a2, _b2)
                              in zip(parts, parts[1:]) if a2 > b1]
                    pieces += [seq[:parts[0][0]], seq[parts[-1][1]:]]
                    for pc in pieces:
                        if feat.location.strand == -1:
                            pc = revcomp(pc)
                        if len(pc) >= min_noncoding and set(pc) <= set("ACGT"):
                            noncoding.append(pc)
    return _dedup(coding), _dedup(noncoding)


def build_sets(fasta_path: str, gff_path: str, min_noncoding: int = 120
               ) -> Tuple[List[str], List[str]]:
    """Return (coding_sequences, noncoding_pieces) from a genome FASTA and a
    GFF3. Coding = spliced CDS passing the filters (terminal stop removed,
    duplicates dropped). Non-coding = introns of the same loci, plus the UTR
    portions of exon features where the GFF3 carries them, each at least
    `min_noncoding` nt. A genome GFF3 has no natural 'flank', so for the
    analysis of the paper use --genbank, which reproduces the paper's sets."""
    try:
        from Bio import SeqIO
    except ImportError:
        sys.exit("[error] biopython is required to read the genome FASTA "
                 "(pip install biopython)")
    genome = {rec.id: str(rec.seq) for rec in SeqIO.parse(fasta_path, "fasta")}
    # tolerate 'chr21' vs '21' style id differences
    alias = {}
    for k in list(genome):
        alias[k] = k
        alias[k.replace("chr", "")] = k
        alias["chr" + k] = k

    def orient(piece, strand):
        return revcomp(piece) if strand == "-" else piece.upper()

    loci = read_gff_loci(gff_path)
    coding, noncoding = [], []
    for loc in loci.values():
        gid = alias.get(loc.seqid)
        if gid is None:
            continue
        chrom = genome[gid]
        cds_seq = _strip_terminal_stop(_splice(chrom, loc.cds, loc.strand))
        if cds_passes(cds_seq):
            coding.append(cds_seq)
        pieces = []
        ex = sorted(loc.exons or loc.cds)
        for (a1, b1), (a2, b2) in zip(ex, ex[1:]):          # introns
            if a2 > b1:
                pieces.append(chrom[b1:a2])
        if loc.exons:                                        # UTR portions
            cs, ce = min(a for a, _ in loc.cds), max(b for _, b in loc.cds)
            for a, b in ex:
                if a < cs:
                    pieces.append(chrom[a:min(b, cs)])
                if b > ce:
                    pieces.append(chrom[max(a, ce):b])
        for pc in pieces:
            pc = orient(pc, loc.strand)
            if len(pc) >= min_noncoding and all(b in "ACGT" for b in pc):
                noncoding.append(pc)
    return _dedup(coding), _dedup(noncoding)


# --------------------------------------------------------------------------- #
# Length-matched sampling and ROC-AUC
# --------------------------------------------------------------------------- #
def roc_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """ROC-AUC via the Mann-Whitney U identity (no sklearn dependency)."""
    from scipy.stats import rankdata
    pos = labels == 1
    n1, n0 = int(pos.sum()), int((~pos).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = rankdata(scores)
    return (r[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


# --------------------------------------------------------------------------- #
# Analyses
# --------------------------------------------------------------------------- #
DEFAULT_BINS = [(40, 60), (60, 100), (100, 150), (150, 250), (250, 400),
                (400, 800), (800, 100000)]
# The Results quote the hexamer comparison on a different binning from the
# Fickett comparison; pass --bins to reproduce either.
HEXAMER_BINS = [(70, 110), (110, 180), (180, 300), (300, 600), (600, 100000)]


def _window_matched_negatives(lengths, noncoding, rng):
    """For each positive length L, draw a non-coding piece at least L nt long
    uniformly at random and cut a window of exactly L nt from a uniformly random
    offset. Positives and negatives are therefore matched exactly on length."""
    nc_len = np.array([len(x) for x in noncoding])
    longest = int(np.argmax(nc_len))
    out = []
    for L in lengths:
        pool = np.flatnonzero(nc_len >= L)
        j = int(rng.choice(pool)) if len(pool) else longest
        st = int(rng.integers(0, max(1, nc_len[j] - L + 1)))
        out.append(noncoding[j][st:st + L])
    return out


def analysis_fig5a(coding, noncoding, rng, bins=None, n_draws=200,
                   n_hexamer_draws=200) -> "list[dict]":
    """AUC by ORF-length bin for the two indices and the two comparators.

    Each coding sequence in a bin is paired with a non-coding window of exactly
    its own length (see _window_matched_negatives). Because a single draw of
    negatives is noisy on bins of ~60-80 sequences, the draw is repeated
    `n_draws` times and each AUC is reported as the mean with a 2.5-97.5%
    interval over draws; the hexamer model, which is refitted in five folds per
    draw, uses the first `n_hexamer_draws` draws."""
    bins = bins or DEFAULT_BINS
    scored = [(len(s), nmsi_frame0(s), nmsi_framefree(s), fickett_asymmetry(s), s)
              for s in coding]
    rows = []
    for lo, hi in bins:
        cod = [c for c in scored if lo <= c[0] // 3 < hi]
        if len(cod) < 10:
            continue
        y = np.array([1] * len(cod) + [0] * len(cod))
        acc = {"frame_resolved": [], "frame_free": [], "fickett": [], "hexamer_outoffold": []}
        for d in range(n_draws):
            neg = _window_matched_negatives([c[0] for c in cod], noncoding, rng)
            ns = [(nmsi_frame0(x), nmsi_framefree(x), fickett_asymmetry(x)) for x in neg]
            fr = np.array([c[1] for c in cod] + [n[0] for n in ns])
            ff = np.array([c[2] for c in cod] + [n[1] for n in ns])
            fk = np.array([c[3] for c in cod] + [n[2] for n in ns])
            # coding shows GREATER imbalance => LOWER nMSI, so score = 1 - nMSI;
            # Fickett's position asymmetry is HIGHER in coding and is used as is.
            acc["frame_resolved"].append(roc_auc(1 - fr, y))
            acc["frame_free"].append(roc_auc(1 - ff, y))
            acc["fickett"].append(roc_auc(fk, y))
            if d < n_hexamer_draws:
                acc["hexamer_outoffold"].append(
                    hexamer_cv_auc([c[4] for c in cod], neg, folds=5, seed=d))
        row = dict(orf_codon_bin=f"{lo}-{hi if hi < 100000 else '+'}",
                   n_coding=len(cod), n_noncoding=len(cod), n_draws=n_draws)
        for k, v in acc.items():
            v = np.asarray(v, float)
            row[f"auc_{k}"] = round(float(np.nanmean(v)), 3)
            row[f"auc_{k}_lo"] = round(float(np.nanpercentile(v, 2.5)), 3)
            row[f"auc_{k}_hi"] = round(float(np.nanpercentile(v, 97.5)), 3)
        row["pct_draws_fickett_ge_index"] = round(
            100 * float(np.mean(np.array(acc["fickett"]) >= np.array(acc["frame_resolved"]))), 1)
        rows.append(row)
    return rows


def analysis_fig5b(coding, noncoding, seed=None) -> dict:
    """Figure 5B: frame-0 index of real CDS, after whole-codon permutation,
    after shuffling bases within each codon position and after a
    mononucleotide shuffle, on all coding sequences (20 shuffles each), with
    codon-permutation invariance (max |delta| over all sequences) and the
    fraction of the depression recovered by the within-position shuffle.
    Definitions: coding_extras.py header. One row."""
    import coding_extras as cx
    return cx.fig5b_codon_structure(coding, noncoding,
                                    seed=cx.SEED_5B if seed is None else seed)


def analysis_fig5b_recovery(row: dict) -> dict:
    """The decomposition of the Figure 5B depression (reference = the
    mononucleotide-shuffled value of the same sequences). The within-position
    shuffle keeps per-position composition and destroys codon identity, so the
    fraction it RECOVERS is due to codon identity beyond per-position
    composition and the fraction that REMAINS is carried by per-position
    composition (what Fickett's statistic reads)."""
    return dict(
        total_depression=row["mean_mononucleotide_shuffled"] - row["mean_real"],
        recovered_by_within_position_frac=row["recovered_by_within_position_frac"],
        codon_identity_beyond_position_frac=row["codon_identity_beyond_position_frac"],
        per_position_composition_frac=row["per_position_composition_frac"],
        recovered_vs_noncoding_frac=row["recovered_vs_noncoding_frac"])


# --------------------------------------------------------------------------- #
def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--genbank", nargs="+", metavar="GB",
                     help="GenBank training files, one gene per LOCUS with flanks "
                          "(the paper: AUGUSTUS D. melanogaster 2L+2R)")
    src.add_argument("--fasta", help="genome FASTA (.fa/.fa.gz); requires --gff")
    ap.add_argument("--gff", help="annotation GFF3 with CDS features (with --fasta)")
    ap.add_argument("--min-noncoding", type=int, default=120,
                    help="minimum length of a non-coding piece in nt (paper: 120)")
    ap.add_argument("--out", default="results/cds", help="output directory")
    ap.add_argument("--bins", choices=["default", "hexamer"], default="default",
                    help="ORF-length binning for Figure 5A: 'default' is the "
                         "binning used for the Fickett comparison, 'hexamer' the "
                         "one used for the hexamer comparison in the Results")
    ap.add_argument("--n-draws", type=int, default=200,
                    help="repeated draws of length-matched negatives for Figure 5A")
    ap.add_argument("--n-hexamer-draws", type=int, default=200,
                    help="draws on which the hexamer model is refitted (slower)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    rng = np.random.default_rng(a.seed)

    if a.genbank:
        sys.stderr.write(f"[read] GenBank {' '.join(a.genbank)}\n")
        coding, noncoding = build_sets_genbank(a.genbank, a.min_noncoding)
    else:
        if not a.gff:
            ap.error("--fasta requires --gff")
        sys.stderr.write(f"[read] genome {a.fasta} + annotation {a.gff}\n")
        coding, noncoding = build_sets(a.fasta, a.gff, a.min_noncoding)
    sys.stderr.write(f"[built] {len(coding)} coding sequences passing filters, "
                     f"{len(noncoding)} non-coding pieces\n")
    if len(coding) < 20 or len(noncoding) < 20:
        sys.exit("[error] too few coding/non-coding sequences — check the "
                 "annotation has CDS features and matches the FASTA seq ids.")

    os.makedirs(a.out, exist_ok=True)
    import pandas as pd

    # ---- Figure 5A
    sys.stderr.write("[run] Figure 5A: AUC by ORF-length bin\n")
    f5a = analysis_fig5a(coding, noncoding, rng,
                         HEXAMER_BINS if a.bins == "hexamer" else DEFAULT_BINS,
                         a.n_draws, a.n_hexamer_draws)
    df = pd.DataFrame(f5a)
    df.to_csv(os.path.join(a.out, f"fig5a_auc_by_orf_length_{a.bins}.csv"), index=False)
    print("\n=== Figure 5A: coding vs non-coding AUC by ORF length ===")
    print(df.to_string(index=False))

    # ---- Extras (each with its own fixed seed, independent of --seed, so
    #      they do not depend on the Figure 5A draws). Definitions are in the
    #      header of coding_extras.py.
    import coding_extras as cx
    pd.set_option("display.width", 200)

    # ---- Figure 5B
    sys.stderr.write("[run] Figure 5B: codon-structure controls\n")
    f5b = analysis_fig5b(coding, noncoding)
    rec = analysis_fig5b_recovery(f5b)
    pd.DataFrame([f5b]).to_csv(os.path.join(a.out, "fig5b_codon_structure.csv"),
                               index=False)
    print("\n=== Figure 5B: frame-0 index under controls (all "
          f"{f5b['n_sequences']} CDS, {f5b['n_shuffles_per_sequence']} shuffles each) ===")
    print(f"  real / codon-permuted / within-position / mononucleotide: "
          f"{f5b['mean_real']:.3f} / {f5b['mean_codon_permuted']:.3f} / "
          f"{f5b['mean_within_position_shuffled']:.3f} / "
          f"{f5b['mean_mononucleotide_shuffled']:.3f}")
    print(f"  max |delta| under whole-codon permutation: "
          f"{f5b['max_abs_delta_codon_permutation_all']:.2f}")
    print(f"  recovered by within-position shuffle (= codon identity beyond "
          f"per-position composition): {100*rec['recovered_by_within_position_frac']:.1f}% "
          f"[{100*f5b['recovered_by_within_position_lo']:.1f}, "
          f"{100*f5b['recovered_by_within_position_hi']:.1f}]; remaining "
          f"(per-position composition): {100*rec['per_position_composition_frac']:.1f}%")
    print(f"  (vs exact-length non-coding windows, mean {f5b['mean_noncoding_window']:.3f}, "
          f"as the reference: {100*rec['recovered_vs_noncoding_frac']:.1f}% recovered)")

    # ---- Frame identification
    sys.stderr.write("[run] frame identification\n")
    fid_rows, Fv = cx.frame_identification(coding)
    fid = pd.DataFrame(fid_rows)
    fid.to_csv(os.path.join(a.out, "frame_identification_by_orf_length.csv"), index=False)
    pd.DataFrame(dict(codons=[len(s) // 3 for s in coding], true_frame=Fv[:, 0], frame1=Fv[:, 1],
                      frame2=Fv[:, 2])).to_csv(os.path.join(a.out, "frame_values.csv"), index=False)
    print("\n=== Frame identification: which frame gives the lowest index ===")
    print(fid.to_string(index=False))

    # ---- Fickett vs symmetry
    sys.stderr.write("[run] Fickett vs symmetry: correlation and logistic CV\n")
    fvs = pd.DataFrame(cx.fickett_vs_symmetry(coding, noncoding, n=660))
    fvs.to_csv(os.path.join(a.out, "fickett_vs_symmetry_logistic.csv"), index=False)
    print("\n=== Fickett vs symmetry (5-fold stratified CV, standardised) ===")
    print(fvs.to_string(index=False))

    print(f"\n[saved] tables written to {a.out}/")
    sys.stderr.write(
        "\n[note] Exact figures depend on the annotated loci supplied; the "
        "paper uses the AUGUSTUS D. melanogaster 2L/2R training loci, and its "
        "Limitations note that replication on human coding sequence is needed.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
