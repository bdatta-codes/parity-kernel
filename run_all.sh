#!/usr/bin/env bash
# Reproduce every result that needs no external reference data, then draw every
# figure. Results that need the reference data (chromosome 21, the AUGUSTUS fly
# loci) or long run times are read from the CSVs in results/; the commands that
# regenerate them are printed at the end and listed in README.md.
# Run time: about 40 minutes on two cores.
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONPATH="$PWD/src"

echo "[1/7] self-tests of the reference implementations"
python3 src/selftest.py                      # MSI/nMSI and G4Hunter reference values
python3 src/coding_comparators.py --selftest # Fickett statistic and hexamer score

echo "[2/7] closed form: Tables 1-3, Supplementary Table S7, Figure 2 data"
python3 src/table1.py                        # Table 1
python3 src/closed_form.py                   # Tables 2 and 3, Supplementary Table S7, Figure 2
python3 src/kappa_inf.py                     # large-N overlap constant (Table 2, Supplementary Methods S10)

echo "[3/7] mutagenesis of 538 sequences and substitution scan"
python3 src/experiment_538.py                # Figures 1C and 3, Supplementary Table S13
python3 src/substitution_scan.py             # Supplementary Table S19, Supplementary Figure S3

echo "[4/7] guanine run lengths and composition (Figure 4)"
python3 src/g4_runlength_analysis.py

echo "[5/7] supplementary tables from the main experiment"
python3 src/si_compute.py                    # Supplementary Tables S2, S6 (checks), S8, S12, S13, S19, S24
python3 src/si_sensitivity.py                # Supplementary Table S10
python3 src/reversible_memory.py             # Supplementary Methods S11
python3 src/wt_vs_own_shuffles.py            # wild types against their own shuffles (Results)
python3 src/si_table_repeats_excluded.py     # Supplementary Table S3 (from results/chr21*/)

echo "[6/7] placement tests, labelled sequences, kernel decomposition, planted features"
python3 src/g4hunter_paired_null.py          # G4Hunter under the paired nulls
python3 src/placement_null.py                # Supplementary Table S15
python3 src/placement_null_variants.py       # Supplementary Table S16
python3 src/consensus_positive_subset.py     # Supplementary Table S9
python3 src/matched_placement.py             # Supplementary Table S17
python3 src/experimental_labels.py           # Supplementary Tables S11 and S23
python3 src/quoted_counts.py                 # counts and tests quoted in the text
python3 src/kernel_decomposition.py          # Supplementary Table S20
python3 src/planted_control.py               # Supplementary Table S26
python3 src/si_additional_tables.py          # Supplementary Tables S4, S5, S9, S15-S17, S20, S23, S25, S26
python3 src/si_a549_full_tables.py           # Supplementary Tables S14 and S21
python3 src/ref21_pairs.py                   # Supplementary Table S1
python3 src/si_composition_and_matching_tables.py   # Supplementary Tables S11, S18 and S22

echo "[7/7] figures"
python3 src/make_figs.py                            # Figures 1 and 3
python3 src/make_fig2.py                            # Figure 2
python3 src/make_fig4.py --out figures/Figure4.png  # Figure 4
python3 src/make_fig5.py --out figures/Figure5.png  # Figure 5
python3 src/make_fig6.py                            # Figure 6
python3 src/si_figs.py                              # Supplementary Figures S1-S3

echo "done: see results/ and figures/"
echo
echo "Read from results/ rather than recomputed (long run times; see README.md):"
echo "  python3 src/a549_full_set.py                    # ~20 min; Figure 4C, Supplementary Table S21"
echo "  python3 src/a549_full_mutagenesis.py --procs 2  # ~60 min; Supplementary Table S14"
echo "  python3 src/composition_share.py                # several hours; Supplementary Table S22"
echo "  python3 src/matched_placement_length.py         # ~2 h; Supplementary Table S18"
echo "Read from results/ because they need reference data (see README.md):"
echo "  bash data/fetch_grch38_chr21.sh && bash data/fetch_augustus_fly.sh"
echo "  python3 src/chromosome21_windows.py --chr21 data/Homo_sapiens.GRCh38.dna.chromosome.21.fa.gz --out results/chr21"
echo "  python3 src/chr21_markov_decomposition.py --chr21 data/Homo_sapiens.GRCh38.dna.chromosome.21.fa.gz"
echo "  python3 src/coding_sequence_analysis.py --genbank data/fly2L.gb data/fly2R.gb --out results/cds"
echo "  python3 src/length_matched.py --genbank data/fly2L.gb data/fly2R.gb --chr21 data/Homo_sapiens.GRCh38.dna.chromosome.21.fa.gz"
echo "  python3 src/kernel_decomposition.py --chr21 data/Homo_sapiens.GRCh38.dna.chromosome.21.fa.gz"
