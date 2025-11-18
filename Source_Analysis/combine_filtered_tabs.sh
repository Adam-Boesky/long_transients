#!/bin/bash

#SBATCH -p test
#SBATCH -c 12                                                                                           # Number of cores (-c)
#SBATCH --mem=32G                                                                                      # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH -t 0-1:00                                                                                      # Runtime in D-HH:MM, minimum of 10 minutes

# Paths to STDOUT or STDERR files should be absolute or relative to current working directory
#SBATCH -o /n/holystore01/LABS/berger_lab/Users/aboesky/long_transients/extraction_logs/myoutput_\%j.out           # File to which STDOUT will be written, %j inserts jobid
#SBATCH -e /n/holystore01/LABS/berger_lab/Users/aboesky/long_transients/extraction_logs/myerrors_\%j.err           # File to which STDERR will be written, %j inserts jobid
#SBATCH --mail-user=aboesky@college.harvard.edu     # Send email to user

# Remember:
# The variable $TMPDIR points to the local hard disks in the computing nodes.
# The variable $HOME points to your home directory.
# The variable $SLURM_JOBID stores the ID number of your job.

# Load modules
#################################
module load python/3.12.5-fasrc01
source activate long_transients2

python3 /n/home04/aboesky/berger/long_transients/Source_Analysis/combine_filtered_tabs.py
