#!/bin/bash

#SBATCH -p shared
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH -t 0-1:00
#SBATCH -o /n/holystore01/LABS/berger_lab/Users/aboesky/long_transients/filtering_logs/combine_filtered_tabs_%j.out
#SBATCH -e /n/holystore01/LABS/berger_lab/Users/aboesky/long_transients/filtering_logs/combine_filtered_tabs_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=aboesky@college.harvard.edu

module load gcc/12.2.0-fasrc01
module load python/3.12.5-fasrc01
source activate long_transients2

cd /n/home04/aboesky/berger/long_transients
python3 -u Source_Analysis/combine_filtered_tabs.py
