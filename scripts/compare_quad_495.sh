#!/bin/bash
#SBATCH -p shared
#SBATCH -c 2
#SBATCH --mem=16G
#SBATCH -t 0-00:30
#SBATCH -o /n/holystore01/LABS/berger_lab/Users/aboesky/long_transients/extraction_logs/compare_quad_495_%j.out
#SBATCH -e /n/holystore01/LABS/berger_lab/Users/aboesky/long_transients/extraction_logs/compare_quad_495_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=aboesky@college.harvard.edu

module load gcc/12.2.0-fasrc01
module load python/3.12.5-fasrc01
source activate long_transients2

cd /n/home04/aboesky/berger/long_transients

python3 -u scripts/test_ps1_local_vs_casjobs.py \
    --ra-center 340.4447 --dec-center 2.1124 --half-width 0.4474 \
    --bands g r i
