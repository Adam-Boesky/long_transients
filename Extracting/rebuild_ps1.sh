#!/bin/bash
#SBATCH -J ps1_rebuild
#SBATCH -p shared
#SBATCH -c 16
#SBATCH --mem=64G
#SBATCH -t 1-12:00
#SBATCH -o /n/holystore01/LABS/berger_lab/Users/aboesky/long_transients/extraction_logs/ps1_rebuild_%j.out
# Rebuild every quadrant PSTARR.hdf5 over the union of its band images (plan_filter_review.md #12). Resumable.
module load gcc/12.2.0-fasrc01
module load python/3.12.5-fasrc01
source activate long_transients2
cd ~/berger/long_transients
python -u -m Extracting.rebuild_ps1_catalogs --quad-list ${QUAD_LIST:-/n/holystore01/LABS/berger_lab/Users/aboesky/long_transients/ps1_rebuild_quads_10_22.csv} --workers ${WORKERS:-16}
