#!/bin/bash
#SBATCH -p shared
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH -t 0-01:00
#SBATCH -o /n/holystore01/LABS/berger_lab/Users/aboesky/long_transients/extraction_logs/test_one_quad_%j.out
#SBATCH -e /n/holystore01/LABS/berger_lab/Users/aboesky/long_transients/extraction_logs/test_one_quad_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=aboesky@college.harvard.edu

module load gcc/12.2.0-fasrc01
module load python/3.12.5-fasrc01
source activate long_transients2

cd /n/home04/aboesky/berger/long_transients

python3 -u -c "
import sys
sys.path.insert(0, '.')
from Extracting.run_extraction import process_quadrant
process_quadrant(fieldid=228, ccdid=9, qid=1, bands=['g', 'r'])
print('Done.')
"
