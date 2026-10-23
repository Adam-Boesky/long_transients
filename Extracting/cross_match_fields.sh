#!/bin/bash
# Re-associate every quadrant of FIELD_LIST against its PS1 catalog, then rebuild the merged field tables.
# Old merged tables are moved to MOVE_OLD_TO first (if set). Used after the PS1 rebuild (plan_filter_review.md #12).

#SBATCH -J xmatch
#SBATCH -p shared
#SBATCH -c 12
#SBATCH --mem=128G
#SBATCH -t 3-00:00
#SBATCH -o /n/holystore01/LABS/berger_lab/Users/aboesky/long_transients/extraction_logs/cross_match_fields_%j.out

module load gcc/12.2.0-fasrc01
module load python/3.12.5-fasrc01
source activate long_transients2

cd /n/home04/aboesky/berger/long_transients
B=/n/holystore01/LABS/berger_lab/Users/aboesky/long_transients
FIELDS=$(cat ${FIELD_LIST:?set FIELD_LIST})

echo "XMATCH associate start $(date) on $(echo $FIELDS | wc -w) fields"
python3 -u -m Extracting.cross_match -aq -ow -nt 12 -f $FIELDS || exit 1

if [ -n "${MOVE_OLD_TO}" ]; then
    echo "XMATCH moving old merged tables to ${MOVE_OLD_TO} $(date)"
    mkdir -p ${MOVE_OLD_TO}
    for f in $FIELDS; do
        for x in $B/catalog_results/field_results/${f}_*.hdf5; do
            [ -f "$x" ] && [ ! -e "${MOVE_OLD_TO}/$(basename $x)" ] && mv "$x" ${MOVE_OLD_TO}/
        done
    done
fi

echo "XMATCH merge start $(date)"
python3 -u -m Extracting.cross_match -mf -ow -nt 12 -f $FIELDS || exit 1
echo "XMATCH_DONE $(date)"
