#!/usr/bin/env python3
"""frame_identification.py -- regenerate only the frame-identification outputs
(results/cds/frame_identification_by_orf_length.csv and frame_values.csv) from
the AUGUSTUS fly GenBank files, without rerunning the rest of
coding_sequence_analysis.py (which writes the same files).

Usage: python3 src/frame_identification.py --genbank data/fly2L.gb data/fly2R.gb [--out results/cds/]
"""
import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pandas as pd
import coding_sequence_analysis as csa, coding_extras as cx

ap = argparse.ArgumentParser(); ap.add_argument("--genbank", nargs="+", required=True)
ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "cds"))
a = ap.parse_args()
coding, _ = csa.build_sets_genbank(a.genbank)
rows, F = cx.frame_identification(coding)
pd.DataFrame(rows).to_csv(os.path.join(a.out, "frame_identification_by_orf_length.csv"), index=False)
pd.DataFrame(dict(codons=[len(s) // 3 for s in coding], true_frame=F[:, 0], frame1=F[:, 1],
                  frame2=F[:, 2])).to_csv(os.path.join(a.out, "frame_values.csv"), index=False)
print(pd.DataFrame(rows).to_string(index=False))
