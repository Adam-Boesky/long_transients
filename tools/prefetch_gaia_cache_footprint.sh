#!/bin/bash

# Rebuild the Gaia cache for the re-extraction fields from their full IRSA footprint,
# moving the old (partial-coverage) caches aside first.

#SBATCH -p shared
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH -t 0-24:00

#SBATCH -o /n/holystore01/LABS/berger_lab/Users/aboesky/long_transients/filtering_logs/prefetch_gaia_fp_%j.out
#SBATCH -e /n/holystore01/LABS/berger_lab/Users/aboesky/long_transients/filtering_logs/prefetch_gaia_fp_%j.err

module load gcc/12.2.0-fasrc01
module load python/3.12.5-fasrc01
source activate long_transients2

B=/n/holystore01/LABS/berger_lab/Users/aboesky/long_transients
python3 -u /n/home04/aboesky/berger/long_transients/scripts/prefetch_gaia_cache.py --workers 3 \
    --footprint $B/gaia_footprint_vertices_164_fields_10_13_2026.csv \
    --move-old-to $B/gaia_cache_pre_reextract ${GAIA_ARGS}
