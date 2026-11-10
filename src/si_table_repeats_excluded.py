#!/usr/bin/env python3
"""si_table_repeats_excluded.py — Supplementary Table S3: chromosome 21 analyses with repeats
included (results/chr21/, main text) and excluded (results/chr21_repeats_excluded/,
from chromosome21_windows.py --mask-repeats on the soft-masked Ensembl file).

License: MIT
"""
import os
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(ROOT, "results")
rows = []
for tag, d in (("repeats_included", "chr21"), ("repeats_excluded", "chr21_repeats_excluded")):
    t4 = pd.read_csv(os.path.join(R, d, "table4_windows_by_length.csv"))
    b = pd.read_csv(os.path.join(R, d, "fig6b_deficit_cpg.csv")).iloc[0]
    t5 = pd.read_csv(os.path.join(R, d, "table5_g4_symmetry.csv")).iloc[0]
    rec = {}
    for _, r in t4.iterrows():
        L = int(r.window_nt)
        rec[f"observed_nmsi_{L}nt"] = r.observed_nmsi
        rec[f"mean_z_{L}nt"] = r.mean_z
        rec[f"n_windows_{L}nt"] = int(r.n)
    rec.update(n_windows_300nt_tiles=int(b.n),
               deficit_vs_gc_rho=b.deficit_vs_gc_rho,
               deficit_vs_cpgunder_rho=b.deficit_vs_cpgunder_rho,
               deficit_vs_cpgunder_partial_gc_rho=b.deficit_vs_cpgunder_partial_gc_rho,
               frac_windows_with_g4=t5.frac_windows_with_g4,
               mean_nmsi_with_g4=t5.mean_nmsi_with_g4,
               mean_nmsi_without_g4=t5.mean_nmsi_without_g4,
               g4_vs_symmetry_rho=t5.g4_vs_symmetry_rho,
               gc_vs_symmetry_rho=t5.gc_vs_symmetry_rho,
               g4_vs_gc_rho=t5.g4_vs_gc_rho,
               g4_vs_symmetry_partial_gc_rho=t5.g4_vs_symmetry_partial_gc_rho)
    rows.append((tag, rec))
T = pd.DataFrame({tag: rec for tag, rec in rows}).rename_axis("quantity").reset_index()
T.to_csv(os.path.join(R, "supplementary_tables", "TableS3_chr21_repeats.csv"), index=False)
print(T.to_string(index=False))
