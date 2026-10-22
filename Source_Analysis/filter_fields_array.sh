#!/bin/bash
# Filter one field per array task: sbatch --array=1-485%100 Source_Analysis/filter_fields_array.sh
# FIELD_LIST: one 6-digit field ID per line (task N filters line N). A field whose output folder exists is skipped.

#SBATCH -J filter_field
#SBATCH -p shared
#SBATCH -c 4
#SBATCH --mem=64G
#SBATCH -t 0-08:00
#SBATCH -o /n/holystore01/LABS/berger_lab/Users/aboesky/long_transients/filtering_logs/filter_field_%A_%a.out

module load gcc/12.2.0-fasrc01
module load python/3.12.5-fasrc01
source activate long_transients2

cd /n/home04/aboesky/berger/long_transients
FIELD_LIST=${FIELD_LIST:-/n/holystore01/LABS/berger_lab/Users/aboesky/long_transients/fields_final_485.txt}
FIELD=$(sed -n "${SLURM_ARRAY_TASK_ID}p" ${FIELD_LIST})
echo "FILTER_FIELD_START ${FIELD} $(date)"
python3 -u -m Source_Analysis.filter_fields --field ${FIELD} && echo "FILTER_FIELD_DONE ${FIELD} $(date)"
