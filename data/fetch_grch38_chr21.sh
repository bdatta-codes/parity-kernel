#!/usr/bin/env bash
# Fetch chromosome 21 of the GRCh38 primary assembly from Ensembl (unmasked
# 'dna' file) for Table 4, Table 5 and Figure 6, and check the sequence.
# The chromosome sequence is identical in every Ensembl release of GRCh38, so
# the check is on the sequence itself (upper-cased, header and newlines
# removed), not on the compressed file.
set -euo pipefail
cd "$(dirname "$0")"
F=Homo_sapiens.GRCh38.dna.chromosome.21.fa.gz
curl -fsSL "https://ftp.ensembl.org/pub/current_fasta/homo_sapiens/dna/${F}" -o "$F"
python3 - "$F" <<'PY'
import gzip, hashlib, sys
s = "".join(l.strip() for l in gzip.open(sys.argv[1], "rt") if not l.startswith(">")).upper()
md5 = hashlib.md5(s.encode()).hexdigest()
assert len(s) == 46709983, f"unexpected length {len(s)}"
assert md5 == "2f45a3455007b7e271509161e52954a9", f"unexpected sequence MD5 {md5}"
print(f"ok: {len(s):,} bp, sequence MD5 {md5}")
PY
