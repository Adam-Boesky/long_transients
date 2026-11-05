#!/bin/bash

#SBATCH -p shared
#SBATCH -c 16
#SBATCH --mem=80G
#SBATCH -t 0-06:00
#SBATCH -o /n/holystore01/LABS/berger_lab/Users/aboesky/long_transients/filtering_logs/kde_prefetch_%j.out
#SBATCH -e /n/holystore01/LABS/berger_lab/Users/aboesky/long_transients/filtering_logs/kde_prefetch_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=aboesky@college.harvard.edu

module load gcc/12.2.0-fasrc01
module load python/3.12.5-fasrc01
source activate long_transients2

cd /n/home04/aboesky/berger/long_transients
python3 -u scripts/prefetch_kde_envelopes.py
