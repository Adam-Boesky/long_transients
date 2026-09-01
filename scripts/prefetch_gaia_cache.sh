#!/bin/bash

#SBATCH -p test
#SBATCH -c 4
#SBATCH --mem=400G
#SBATCH -t 0-04:00

#SBATCH -o /n/holystore01/LABS/berger_lab/Users/aboesky/long_transients/filtering_logs/prefetch_gaia_%j.out
#SBATCH -e /n/holystore01/LABS/berger_lab/Users/aboesky/long_transients/filtering_logs/prefetch_gaia_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=aboesky@college.harvard.edu

module load python/3.12.5-fasrc01
source activate long_transients2

python3 -u /n/home04/aboesky/berger/long_transients/scripts/prefetch_gaia_cache.py
