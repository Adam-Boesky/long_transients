#!/bin/bash
#SBATCH -p shared
#SBATCH -c 2
#SBATCH --mem=40G
#SBATCH -t 0-01:00
#SBATCH -o /n/holystore01/LABS/berger_lab/Users/aboesky/long_transients/filtering_logs/kde_field_%A_%a.out
#SBATCH -e /n/holystore01/LABS/berger_lab/Users/aboesky/long_transients/filtering_logs/kde_field_%A_%a.err

TODO=/n/holystore01/LABS/berger_lab/Users/aboesky/long_transients/kde_todo.txt
FIELD=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$TODO")

if [ -z "$FIELD" ]; then
    echo "ERROR: no field at array index $SLURM_ARRAY_TASK_ID"
    exit 1
fi

echo "Processing field $FIELD (array task $SLURM_ARRAY_TASK_ID)..."

cd /n/home04/aboesky/berger/long_transients
python3 -u scripts/prefetch_kde_envelopes.py "$FIELD"
