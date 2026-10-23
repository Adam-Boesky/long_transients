#!/bin/bash
# Generate the list of fields that still need KDE caches, then submit one
# SLURM array job per field (up to 5 running at a time on the test partition).

cd /n/home04/aboesky/berger/long_transients

module load gcc/12.2.0-fasrc01
module load python/3.12.5-fasrc01
source activate long_transients2

TODO=/n/holystore01/LABS/berger_lab/Users/aboesky/long_transients/kde_todo.txt

python3 - <<'EOF'
import os, numpy as np
DATA_PATH = '/n/holystore01/LABS/berger_lab/Users/aboesky/long_transients'
BANDS = ('g', 'r', 'i')
def cached(f):
    d = os.path.join(DATA_PATH, 'kde_envelopes', f)
    return all(os.path.exists(os.path.join(d, f'{b}.npz')) for b in BANDS)
g = np.load(os.path.join(DATA_PATH, 'g_imaged_fields.npy'))
r = np.load(os.path.join(DATA_PATH, 'r_imaged_fields.npy'))
i = np.load(os.path.join(DATA_PATH, 'i_imaged_fields.npy'))
fields = [str(f).zfill(6) for f in sorted(np.intersect1d(np.intersect1d(g, r), i))]
todo = [f for f in fields if not cached(f)]
print(f'{len(todo)} fields need KDE pre-population')
with open('/n/holystore01/LABS/berger_lab/Users/aboesky/long_transients/kde_todo.txt', 'w') as fh:
    fh.write('\n'.join(todo))
EOF

N=$(wc -l < "$TODO")
if [ "$N" -eq 0 ]; then
    echo "All KDE caches already built — nothing to do."
    exit 0
fi

echo "Submitting array of $N jobs (indices 0-$((N-1)), max 5 concurrent)..."
sbatch --array=0-$((N-1))%5 scripts/prefetch_kde_one_field.sh
