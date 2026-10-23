#!/bin/bash
#SBATCH -p shared
#SBATCH -c 2
#SBATCH --mem=16G
#SBATCH -t 0-01:00
#SBATCH -o /n/holystore01/LABS/berger_lab/Users/aboesky/long_transients/filtering_logs/repair_000788_000859_%j.out
#SBATCH -e /n/holystore01/LABS/berger_lab/Users/aboesky/long_transients/filtering_logs/repair_000788_000859_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=aboesky@college.harvard.edu

module load gcc/12.2.0-fasrc01 python/3.12.5-fasrc01
source activate long_transients2

cd /n/home04/aboesky/berger/long_transients

echo "=== Step 1: Rebuild PSTARR.hdf5, re-crossmatch, re-merge ==="
python3 -u scripts/repair_000788_000859.py

echo "=== Step 2: Re-run filter for 000788 and 000859 ==="
python3 -u - <<'EOF'
import sys
sys.path.insert(0, '/n/home04/aboesky/berger/long_transients')
from Source_Analysis.filter_fields import filter_field

for field in ['000788', '000859']:
    print(f'\nFiltering field {field}...')
    filter_field(field, overwrite=True)
    print(f'Done filtering {field}')
EOF

echo "=== Repair job complete ==="
