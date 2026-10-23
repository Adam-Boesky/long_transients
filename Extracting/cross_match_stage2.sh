#!/bin/bash
# Stage 2 after the Oct 2026 re-extraction: associate only quadrants missing *_associated.hdf5, then rebuild
# the merged field tables of the affected fields (old tables moved to field_results_pre_stage2/ first).

#SBATCH -p shared
#SBATCH -c 12
#SBATCH --mem=128G
#SBATCH -t 3-00:00
#SBATCH -o /n/holystore01/LABS/berger_lab/Users/aboesky/long_transients/extraction_logs/stage2_%j.out
#SBATCH -e /n/holystore01/LABS/berger_lab/Users/aboesky/long_transients/extraction_logs/stage2_err_%j.err

module load gcc/12.2.0-fasrc01
module load python/3.12.5-fasrc01
source activate long_transients2

cd /n/home04/aboesky/berger/long_transients
B=/n/holystore01/LABS/berger_lab/Users/aboesky/long_transients
FIELDS=$(cat ${FIELD_LIST:-$B/stage2_fields_10_20.txt})

echo "STAGE2 associate start $(date)"
python3 -u -m Extracting.cross_match -aq -nt 12 -f $FIELDS || exit 1

echo "STAGE2 moving old merged tables $(date)"
mkdir -p $B/catalog_results/field_results_pre_stage2
for f in $FIELDS; do
    for x in $B/catalog_results/field_results/${f}_*.hdf5; do
        [ -f "$x" ] && [ ! -e "$B/catalog_results/field_results_pre_stage2/$(basename $x)" ] && mv "$x" $B/catalog_results/field_results_pre_stage2/
    done
done

echo "STAGE2 merge start $(date)"
python3 -u -m Extracting.cross_match -mf -ow -nt 12 -f $FIELDS || exit 1
echo "STAGE2_DONE $(date)"
