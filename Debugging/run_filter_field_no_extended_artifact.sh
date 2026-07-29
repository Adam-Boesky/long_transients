#!/bin/bash

#SBATCH -p test
#SBATCH -c 4                                                                                            # Number of cores (-c)
#SBATCH --mem=8G                                                                                        # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH -t 0-00:30                                                                                      # Runtime in D-HH:MM, minimum of 10 minutes

#SBATCH -o /n/home04/aboesky/berger/long_transients/Extracting/extraction_logs/no_artifact_filter_\%j.out # File to which STDOUT will be written, %j inserts jobid
#SBATCH -e /n/home04/aboesky/berger/long_transients/Extracting/extraction_logs/no_artifact_filter_\%j.err # File to which STDERR will be written, %j inserts jobid
#SBATCH --mail-user=aboesky@college.harvard.edu     # Send email to user

# Usage: sbatch run_filter_field_no_extended_artifact.sh [field_name]  (defaults to 000326)
FIELD="${1:-000326}"

module load python/3.12.5-fasrc01
source activate long_transients2

python3 /n/home04/aboesky/berger/long_transients/Debugging/filter_field_no_extended_artifact.py "$FIELD"
