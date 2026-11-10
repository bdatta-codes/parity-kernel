#!/usr/bin/env python3
"""selftest.py -- checks of the reference implementations before any analysis.

  * MSI and nMSI (parity.msi_scores) against the values of the fourteen
    oncogene-promoter quadruplexes (Supplementary Table S12; ref. 21).
  * G4Hunter (parity.g4hunter) against its published reference values for the
    human telomeric repeat and the c-MYC Pu27 motif (Bedrat et al., 2016).

The full set of validation cases (Supplementary Table S6) is run by
si_compute.py.

    python3 src/selftest.py        # exits with status 1 on any failure
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from parity import msi_scores, g4hunter  # noqa: E402

PROMOTERS = [   # name, sequence, MSI, nMSI
    ("c-MYC",  "TGGGGAGGGTGGGGAGGGTGGGGAAGG", 0.973, 0.923),
    ("VEGF",   "GGGGCGGGCCGGGGGCGGGG", 1.000, 1.000),
    ("BCL2",   "GAAGGAGGGCAGGAGGGCTCTGGGTGGGTCTGT", 0.810, 0.600),
    ("HIF-1a", "GCGCGGGGAGGGGAGAGGGGGCGGGAGCGCG", 0.939, 0.667),
    ("KRAS",   "AGGGCGGTGTGGGAAGAGGGAAGAGGGGGAGG", 0.978, 0.933),
    ("BCR",    "GGGCAGGGTGTGGGGAAACAGGG", 0.774, 0.364),
    ("c-KIT",  "AGGGAGGGCGCTGGGAGGAGGG", 0.828, 0.545),
    ("MET",    "CAGGGCAGCGCGCGTGTGGGAAGGGGCGGAGGGAG", 0.857, 0.588),
    ("CARD11", "GGTGGGGTGGGAGTGGGGACAGGGCCGGGGGTGGGGGTGCTTGGGGACA", 0.870, 0.640),
    ("ACC1",   "GCTGGGCGGGCGGGCGGGCGGGTGG", 0.909, 0.769),
    ("HRAS",   "CGGGGCGGGGCGGGGGCGGGGGCG", 1.000, 1.000),
    ("ATG7",   "GGGGCTGGGGTCCCTTGGGGAACTGTATTGGG", 0.650, 0.300),
    ("MDM2",   "GTTCGTGGCTGGGGGCTCGGGGCGCGGGGCGCGGGGCATGGGGC", 0.787, 0.435),
    ("ARID1A", "GGGAGGGGGCGGGGAGGG", 1.000, 1.000),
]
G4HUNTER = [    # name, sequence, published score
    ("human telomeric repeat", "GGGTTAGGGTTAGGGTTAGGG", 1.71),
    ("c-MYC Pu27", "TGGGGAGGGTGGGGAGGGTGGGGAAGG", 2.59),
]


def main() -> int:
    ok = True
    for name, seq, e_msi, e_nmsi in PROMOTERS:
        msi, nmsi = msi_scores(seq)
        if abs(msi - e_msi) > 0.0015 or abs(nmsi - e_nmsi) > 0.0015:
            ok = False
            print(f"[FAIL] {name}: MSI {msi:.3f} (expected {e_msi}), nMSI {nmsi:.3f} (expected {e_nmsi})")
    print(f"[selftest] MSI/nMSI, 14 promoter quadruplexes: {'OK' if ok else 'FAILED'}")
    g_ok = True
    for name, seq, ref in G4HUNTER:
        v = g4hunter(seq)
        if abs(v - ref) > 0.005:
            g_ok = False
            print(f"[FAIL] G4Hunter {name}: {v:.3f} (published {ref})")
    print(f"[selftest] G4Hunter reference values: {'OK' if g_ok else 'FAILED'}")
    ok &= g_ok
    print("[selftest] ALL PASSED" if ok else "[selftest] FAILURES PRESENT")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
