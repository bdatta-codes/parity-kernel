"""Rebuild data/a549_grich_300.txt from the A549 G4P-ChIP-seq peak table.

Selection rule (Methods): the peak table of Zheng et al. (Nucleic Acids Res.
2020, 48, 11706-11720; the 43,752 peaks with at least fivefold enrichment, hg19) annotates in each peak every
canonical motif of four or more guanine tracts (G>=3) separated by loops of
at most seven nucleotides (column '4G motif', comma-separated, tracts in
square brackets).  Taking these motifs in peak-table order, keeping those on
the guanine strand (entries beginning '[G') and skipping cytosine-strand
entries, the A549 G-rich set is the first 300, written in upper case.  They
come from 160 peaks on chromosome 1 (A549-chip_peak_3 to A549-chip_peak_1006).

The peak table is not redistributed; supply it as .xlsx or .csv:
    python src/derive_a549_set.py PEAK_TABLE [--check data/a549_grich_300.txt]
With --all OUT, every guanine-strand motif (23,045) is also written, with its
peak and chromosome, to OUT (data/a549_g4p_all_motifs.txt; used by
src/a549_full_set.py).  Not called by run_all.sh.
"""
import argparse, re, sys
import pandas as pd


def guanine_strand_motifs(table):
    for _, row in table.iterrows():
        cell = row["4G motif"]
        if not isinstance(cell, str) or cell.strip() in ("", "."):
            continue
        for entry in cell.split(","):
            entry = entry.strip()
            if entry.startswith("[G"):
                yield row["Peak"], row["chr"], re.sub(r"[\[\]]", "", entry).upper()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("peak_table")
    ap.add_argument("-n", type=int, default=300)
    ap.add_argument("--check", default=None)
    ap.add_argument("--all", metavar="OUT", default=None,
                    help="also write every guanine-strand motif (peak, chr, sequence) to OUT")
    a = ap.parse_args()
    t = (pd.read_csv(a.peak_table) if a.peak_table.endswith(".csv")
         else pd.read_excel(a.peak_table))
    motifs = list(guanine_strand_motifs(t))
    if a.all:
        with open(a.all, "w") as fh:
            fh.write("# index\tpeak\tchr\tsequence\n")
            fh.write("".join(f"{i}\t{p}\t{c}\t{q}\n" for i, (p, c, q) in enumerate(motifs, 1)))
    chosen = motifs[: a.n]
    print(f"{len(motifs)} guanine-strand motifs in table; first {a.n} from "
          f"{len({p for p, _, _ in chosen})} peaks on {sorted({c for _, c, _ in chosen})}, "
          f"{chosen[0][0]} to {chosen[-1][0]}, "
          f"{min(len(s) for *_, s in chosen)}-{max(len(s) for *_, s in chosen)} nt", file=sys.stderr)
    seqs = [s for *_, s in chosen]
    if a.check:
        ref = [l.split("\t")[-1].strip().upper() for l in open(a.check)
               if l.strip() and not l.startswith(("#", ">"))]
        ok = ref == seqs
        print("matches", a.check, ":", ok)
        sys.exit(0 if ok else 1)
    sys.stdout.write("".join(f"{i}\t{s}\n" for i, s in enumerate(seqs, 1)))


if __name__ == "__main__":
    main()
