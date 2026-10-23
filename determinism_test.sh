#!/bin/bash
#SBATCH -p shared
#SBATCH -c 8
#SBATCH --mem=100G
#SBATCH -t 0-04:00
#SBATCH -o /n/holystore01/LABS/berger_lab/Users/aboesky/long_transients/filtering_logs/determ_%j.out
#SBATCH -e /n/holystore01/LABS/berger_lab/Users/aboesky/long_transients/filtering_logs/determ_%j.err
module load gcc/12.2.0-fasrc01
module load python/3.12.5-fasrc01
source activate long_transients2
cd /n/home04/aboesky/berger/long_transients
python3 -u /n/home04/aboesky/berger/long_transients/determinism_test.py "$1"
