#!/bin/bash

#SBATCH -p shared
#SBATCH -c 16                                                                                          # Number of cores (-c)
#SBATCH --mem=160G                                                                                     # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH -t 3-00:00                                                                                     # Runtime in D-HH:MM, minimum of 10 minutes

# Paths to STDOUT or STDERR files should be absolute or relative to current working directory
#SBATCH -o /n/holystore01/LABS/berger_lab/Users/aboesky/long_transients/filtering_logs/myoutput_\%j.out           # File to which STDOUT will be written, %j inserts jobid
#SBATCH -e /n/holystore01/LABS/berger_lab/Users/aboesky/long_transients/filtering_logs/myerrors_\%j.err           # File to which STDERR will be written, %j inserts jobid
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=aboesky@college.harvard.edu     # Send email to user

# Remember:
# The variable $TMPDIR points to the local hard disks in the computing nodes.
# The variable $HOME points to your home directory.
# The variable $SLURM_JOBID stores the ID number of your job.

# Load modules
#################################
module load gcc/12.2.0-fasrc01
module load python/3.12.5-fasrc01
source activate long_transients2

cd /n/home04/aboesky/berger/long_transients
# Args are forwarded, so `sbatch Source_Analysis/filter_fields.sh --descending` runs a
# second job from the far end of the field list to meet an ascending one in the middle.
python3 -u -m Source_Analysis.filter_fields "$@"
