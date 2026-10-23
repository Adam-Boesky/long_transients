#!/bin/bash

#SBATCH -p shared
#SBATCH -c 2
#SBATCH --mem=8G
#SBATCH -t 3-00:00
#SBATCH -o /n/holystore01/LABS/berger_lab/Users/aboesky/long_transients/extraction_logs/prefetch_%j.out
#SBATCH -e /n/holystore01/LABS/berger_lab/Users/aboesky/long_transients/extraction_logs/prefetch_err_%j.err

module load gcc/12.2.0-fasrc01
module load python/3.12.5-fasrc01
source activate long_transients2

cd /n/home04/aboesky/berger/long_transients
B=/n/holystore01/LABS/berger_lab/Users/aboesky/long_transients
LISTS=${LISTS:-"$B/reextract_quadrants_10_13_2026_part1.csv $B/reextract_quadrants_10_13_2026_part2.csv $B/reextract_quadrants_10_13_2026_part3.csv"}
python3 -u -m Extracting.prefetch_deep_images ${LISTS} --threads ${THREADS:-6} ${PREFETCH_ARGS}
