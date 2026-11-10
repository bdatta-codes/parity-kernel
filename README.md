# parity-kernel

Code, data and result tables for

**A Closed-Form Expectation for Intra-Strand Parity Indices and the Limits of Their Use as G-Quadruplex Descriptors**
Arjun Arya, Heli Pancholi and Bhaskar Datta, Indian Institute of Technology Gandhinagar

An intra-strand parity index compares the frequency of a short word with that of its exact reverse on the same strand. The repository contains the closed-form expectation of such an index, the in silico mutagenesis of 538 putative quadruplex sequences, and the code behind every table and figure of the paper and its Supplementary Information. Every number in the paper is produced by a script in `src/`, and every table is also given as a CSV file in `results/`.

---

## 1. Requirements and installation

Python 3.10 or later with NumPy, SciPy, pandas, matplotlib, scikit-learn, Biopython, openpyxl and Pillow (`requirements.txt`). The results in `results/` were produced with Python 3.13, NumPy 2.5.3, SciPy 1.18.1, pandas 3.0.5, matplotlib 3.11.2 and scikit-learn 1.9.1; with these versions a rerun reproduces every CSV byte for byte.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## 2. Quick checks (about one minute)

```bash
export PYTHONPATH=src
python3 src/selftest.py                      # MSI/nMSI and G4Hunter against reference values
python3 src/coding_comparators.py --selftest # Fickett statistic and hexamer score

# the expected nMSI of a 300-nt sliding window at uniform composition (Eq. 4)
python3 -c "import parity; print(parity.nmsi_baseline_exact(298) + 0.615/298**0.5)"
```

## 3. Reproducing the paper

```bash
bash run_all.sh          # about 40 minutes on two cores
```

`run_all.sh` recomputes every result that needs no external reference data and redraws every figure. Results that need the reference data (human chromosome 21 and the *D. melanogaster* loci, which are not redistributed) or several hours of computation are read from the CSV files in `results/`; Section 5 gives the commands that regenerate them. All random seeds are fixed in the scripts.

## 4. Where each result comes from

### Main text

| Item | Script | Output |
|---|---|---|
| Table 1 | `table1.py` | `results/table1_closed_form.csv` |
| Table 2 | `closed_form.py`, `kappa_inf.py` | `results/closed_form/table2_kappa.csv`, `kappa_inf.csv` |
| Table 3 | `closed_form.py` | `results/closed_form/table3_fixed_criterion.csv` |
| Table 4 | `chromosome21_windows.py` * | `results/chr21/table4_windows_by_length.csv` |
| Table 5 | `chromosome21_windows.py` * | `results/chr21/table5_g4_symmetry.csv` |
| Figure 1 | `experiment_538.py`, then `make_figs.py` | `figures/Figure1.png` |
| Figure 2 | `closed_form.py`, then `make_fig2.py` | `figures/Figure2.png` (data: `results/closed_form/`) |
| Figure 3 | `experiment_538.py`, then `make_figs.py` | `figures/Figure3.png` (data: `results/experiment_538_results.csv`, `results/pairs_*.csv`) |
| Figure 4 | `g4_runlength_analysis.py`, `a549_full_set.py` †, then `make_fig4.py` | `figures/Figure4.png` (data: `results/g4/fig4a_*.csv`, `fig4b_*.csv`; panel C: `results/a549_full/per_sequence.csv`) |
| Figure 5 | `coding_sequence_analysis.py` *, then `make_fig5.py` | `figures/Figure5.png` (data: `results/cds/fig5*.csv`) |
| Figure 6 | `chromosome21_windows.py` *, then `make_fig6.py` | `figures/Figure6.png` (data: `results/chr21/`) |
| Numbers quoted in the text | `quoted_counts.py`, `si_compute.py`, `wt_vs_own_shuffles.py`, `reversible_memory.py`, `g4_runlength_analysis.py` | `results/si/*.json`, `results/g4/summary_runlength_three_effects.csv` |

### Supplementary Information

All Supplementary Tables are written to `results/supplementary_tables/TableS<n>_*.csv`, except Tables S6 and S7.

| Item | Script(s) |
|---|---|
| Table S1, pairs of ref. 21 against their paired nulls | `ref21_pairs.py` |
| Table S2, sequence collections | `si_compute.py` |
| Table S3, chromosome 21 with repeats excluded | `chromosome21_windows.py --mask-repeats` *, then `si_table_repeats_excluded.py` |
| Table S4, chromosome 21 windows matched on GC content | `length_matched.py` *, then `si_additional_tables.py` |
| Table S5, decomposition of the chromosome 21 deficit | `chr21_markov_decomposition.py` *, then `si_additional_tables.py` |
| Table S6, validation of each component | `selftest.py`, `coding_comparators.py --selftest`, `si_compute.py` (`results/si/si_summary.json`) |
| Table S7, word length and pairing | `closed_form.py` (`results/closed_form/tableS7_word_length.csv`), `kappa_inf.py` |
| Table S8, verification of abolition | `si_compute.py` |
| Table S9, consensus-positive wild types | `consensus_positive_subset.py`, then `si_additional_tables.py` |
| Table S10, sensitivity of the exact tails | `si_sensitivity.py` |
| Table S11, composition baselines of the labelled sequences | `experimental_labels.py`, then `si_composition_and_matching_tables.py` |
| Table S12, promoter quadruplexes against three baselines | `si_compute.py` |
| Table S13, all conventions and measures | `experiment_538.py`, `g4hunter_paired_null.py`, `si_compute.py` |
| Table S14, mutagenesis and placement test on all A549 motifs | `a549_full_mutagenesis.py` †, then `si_a549_full_tables.py` |
| Table S15, unmatched placement test | `placement_null.py`, then `si_additional_tables.py` |
| Table S16, variants of the unmatched placement test | `placement_null_variants.py`, then `si_additional_tables.py` |
| Table S17, placement-matched test | `matched_placement.py`, then `si_additional_tables.py` |
| Table S18, matched also on tract length and flanks | `matched_placement_length.py` †, then `si_composition_and_matching_tables.py` |
| Table S19, single-substitution scan | `substitution_scan.py`, `si_compute.py` |
| Table S20, kernel decomposition | `kernel_decomposition.py` (control column with `--chr21` *), then `si_additional_tables.py` |
| Table S21, A549 analyses on all motifs | `a549_full_set.py` †, then `si_a549_full_tables.py` |
| Table S22, correlation with G4Hunter under single shuffles | `composition_share.py` †, then `si_composition_and_matching_tables.py` |
| Table S23, experimentally labelled folding | `experimental_labels.py`, then `si_additional_tables.py` |
| Table S24, five published wild-type/mutant pairs | `si_compute.py` |
| Table S25, coding sequence at quadruplex lengths | `length_matched.py` *, then `si_additional_tables.py` |
| Table S26, planted features | `planted_control.py`, then `si_additional_tables.py` |
| Figures S1–S3 | `si_figs.py` |

\* needs the reference data of Section 5. † long run time (Section 5). Intermediate results of the placement tests, labelled sequences, kernel decomposition, planted features and chromosome 21 decomposition are in `results/additional/`.

## 5. Results that need reference data or long run times

```bash
# reference data (not redistributed); both scripts check what they download
bash data/fetch_grch38_chr21.sh   # Ensembl GRCh38 chromosome 21, unmasked
bash data/fetch_augustus_fly.sh   # AUGUSTUS D. melanogaster 2L/2R training loci

# Tables 4 and 5, Figure 6 (about 15 min)
python3 src/chromosome21_windows.py --chr21 data/Homo_sapiens.GRCh38.dna.chromosome.21.fa.gz --out results/chr21
python3 src/make_fig6.py

# Supplementary Table S3: the same analyses with repeats excluded, on the
# soft-masked Ensembl file (same sequence, repeats in lower case)
curl -fsSL -o data/Homo_sapiens.GRCh38.dna_sm.chromosome.21.fa.gz \
    https://ftp.ensembl.org/pub/current_fasta/homo_sapiens/dna/Homo_sapiens.GRCh38.dna_sm.chromosome.21.fa.gz
python3 src/chromosome21_windows.py --chr21 data/Homo_sapiens.GRCh38.dna_sm.chromosome.21.fa.gz \
    --mask-repeats --out results/chr21_repeats_excluded
python3 src/si_table_repeats_excluded.py

# Supplementary Table S5 (about 20 min) and the control column of Table S20
python3 src/chr21_markov_decomposition.py --chr21 data/Homo_sapiens.GRCh38.dna.chromosome.21.fa.gz
python3 src/kernel_decomposition.py --chr21 data/Homo_sapiens.GRCh38.dna.chromosome.21.fa.gz

# The window and deficit-decomposition analyses run on any chromosome FASTA: run
# chromosome21_windows.py --out results/chrNN first, then pass its window file to
# the decomposition, e.g. for chromosome 22
#   python3 src/chromosome21_windows.py --chr21 data/Homo_sapiens.GRCh38.dna.chromosome.22.fa.gz --out results/chr22
#   python3 src/chr21_markov_decomposition.py --chr21 data/Homo_sapiens.GRCh38.dna.chromosome.22.fa.gz --windows results/chr22/windows300.csv.gz --outdir results/chr22 --tag chr22
# (--outdir/--tag keep chromosome 22's outputs in results/chr22/ rather than overwriting chromosome 21's;
#  the chromosome 22 pilot quoted in the Discussion is already provided, pre-computed, in results/chr22/.)

# Figure 5 and the coding-sequence results (about 25 min each); the Results quote
# the Fickett comparison on the default ORF-length bins and the hexamer comparison
# on the hexamer bins
python3 src/coding_sequence_analysis.py --genbank data/fly2L.gb data/fly2R.gb --out results/cds --bins default
python3 src/coding_sequence_analysis.py --genbank data/fly2L.gb data/fly2R.gb --out results/cds --bins hexamer
python3 src/make_fig5.py --out figures/Figure5.png

# Supplementary Tables S4 and S25
python3 src/length_matched.py --genbank data/fly2L.gb data/fly2R.gb \
    --chr21 data/Homo_sapiens.GRCh38.dna.chromosome.21.fa.gz
python3 src/si_additional_tables.py

# the A549 analyses on all 23,040 motifs (Figure 4C; Supplementary Tables S14, S18, S21, S22)
python3 src/a549_full_set.py                    # about 20 min
python3 src/a549_full_mutagenesis.py --procs 2  # about 60 min
python3 src/matched_placement_length.py         # about 2 h (pooled set and A549 motifs)
python3 src/composition_share.py                # several hours
python3 src/si_a549_full_tables.py && python3 src/si_composition_and_matching_tables.py
```

The soft-masking of the Ensembl file can change between releases; the file analysed has 46,709,983 bp of which 22,752,430 are soft-masked (case-sensitive sequence MD5 `a71f3a49fb8699761051eab5591c9f16`).

## 6. Repository layout

| Path | Contents |
|---|---|
| `src/parity.py` | Reference implementation of MSI and nMSI, G4Hunter, the closed-form baseline (`nmsi_baseline_exact`), its large-N form (`nmsi_baseline_asymptotic`) and the exact expectation for non-overlapping counts (`nmsi_expectation_conditional`) |
| `src/closed_form.py` | The closed form for any word length, pairing and composition, the overlap constant κ and its large-N value (`kappa_inf`), and a vectorised simulator |
| `src/shuffles.py` | Mononucleotide shuffle and Altschul–Erickson dinucleotide shuffle |
| `src/experiment_538.py` | The eight mutagenesis conventions, verification of abolition, the paired per-pair null and the exact three-outcome tail |
| `src/chromosome21_windows.py`, `src/chr21_markov_decomposition.py` | Chromosome 21 analyses |
| `src/coding_sequence_analysis.py`, `src/coding_extras.py`, `src/coding_comparators.py` | Coding-sequence analyses and the Fickett and hexamer comparators |
| `src/g4_runlength_analysis.py`, `src/a549_full_set.py`, `src/composition_share.py` | Guanine run lengths, real sequence against its shuffles, composition |
| `src/placement_null*.py`, `src/matched_placement*.py`, `src/consensus_positive_subset.py`, `src/g4hunter_paired_null.py` | Placement tests |
| `src/kernel_decomposition.py`, `src/planted_control.py`, `src/length_matched.py`, `src/experimental_labels.py`, `src/ref21_pairs.py`, `src/frame_identification.py`, `src/reversible_memory.py`, `src/wt_vs_own_shuffles.py` | Further analyses of the Results and Supplementary Information |
| `src/si_*.py`, `src/quoted_counts.py` | Supplementary tables and quoted numbers from the result files |
| `src/make_fig*.py`, `src/si_figs.py` | Figures |
| `src/derive_a549_set.py` | Rebuilds the A549 motif sets from the published peak table |
| `data/` | Sequence sets analysed (below) and the scripts that fetch the reference data |
| `results/` | Every result table as CSV (`supplementary_tables/` holds the Supplementary Tables) |
| `figures/` | Figures 1–6 and Supplementary Figures S1–S3 |

Each script begins with a description of what it computes, its definitions, seeds and outputs.

## 7. Data

* `data/298_sequences_with_MSI_nMSI.xlsx`: the 298 G4-positive sequences of the G4Hunter reference set (Bedrat, Lacroix and Mergny, *Nucleic Acids Res.* 2016, 44, 1746–1759), with G4Hunter scores and with MSI and nMSI values computed with the original implementation of the indices, used to check the reference implementation.
* `data/a549_g4p_all_motifs.txt`: every guanine-strand canonical motif (four or more G≥3 tracts, loops of at most seven nucleotides) annotated in the A549 G4P-ChIP-seq peak table of Zheng *et al.* (*Nucleic Acids Res.* 2020, 48, 11706–11720; 43,752 peaks with at least fivefold enrichment, hg19), in table order: 23,045 motifs (23,040 with defined nMSI) from 14,005 peaks on all 24 chromosomes. `data/a549_grich_300.txt` holds its first 300 entries (the A549 G-rich set). The peak table is not redistributed; `src/derive_a549_set.py PEAK_TABLE --check data/a549_grich_300.txt` rebuilds and checks the G-rich set, and `src/derive_a549_set.py PEAK_TABLE --all data/a549_g4p_all_motifs.txt` writes the full set.
* `data/promoter_panel.fasta`: the fourteen oncogene-promoter quadruplexes; `data/published_pairs.fasta`: the published wild-type/mutant pairs (MYC series, GPX4, CLIC4); `data/ref21_wt_mutant_pairs.tsv`: the fourteen wild-type/mutant pairs of ref. 21 (Arya and Datta, bioRxiv 2026, doi:10.64898/2026.07.05.736592).
* Human chromosome 21 (Ensembl GRCh38, 46,709,983 bp, sequence MD5 `2f45a3455007b7e271509161e52954a9`) and the AUGUSTUS *D. melanogaster* 2L/2R training loci (pinned commit, MD5-checked) are fetched by the scripts in `data/`.

## 8. Licence and citation

Code: MIT Licence (`LICENSE`). Result tables and figures: CC BY 4.0. Sequence data remain subject to the terms of their original sources. Please cite the paper (`CITATION.cff`).
