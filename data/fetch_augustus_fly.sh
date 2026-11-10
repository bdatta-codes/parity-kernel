#!/usr/bin/env bash
# Fetch the annotated loci used for the coding-sequence analyses (Figures 5A,
# 5B): the AUGUSTUS training GenBank files for Drosophila melanogaster
# chromosomes 2L and 2R (Stanke & Waack 2003, Bioinformatics 19 Suppl 2:ii215),
# from the AUGUSTUS repository at a pinned commit. Writes data/fly2L.gb and
# data/fly2R.gb and checks their MD5 sums.
set -euo pipefail
COMMIT=220e5a63c0fd546563472ec6d1f9271faddaf569
RAW="https://raw.githubusercontent.com/Gaius-Augustus/Augustus/${COMMIT}"
cd "$(dirname "$0")"
curl -fsSL "${RAW}/examples/chr2L/genes.gb.train"  -o fly2L.train.gb
curl -fsSL "${RAW}/examples/chr2L/genes.gb.test"   -o fly2L.test.gb
curl -fsSL "${RAW}/docs/tutorial/results/genes.gb" -o fly2R.gb
cat fly2L.train.gb fly2L.test.gb > fly2L.gb && rm fly2L.train.gb fly2L.test.gb
md5sum -c - <<'SUMS'
d45b4175f666ccd7dbb3748cb396655a  fly2L.gb
7d689170540f02a7651f432fc8f11e7a  fly2R.gb
SUMS
echo "fetched: data/fly2L.gb (492 loci), data/fly2R.gb (586 loci)"
