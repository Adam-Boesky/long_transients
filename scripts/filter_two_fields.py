"""Re-run filter for fields 000788 and 000859 after extraction repair."""
import sys
sys.path.insert(0, '/n/home04/aboesky/berger/long_transients')

from Source_Analysis.filter_fields import filter_field

for field in ['000788', '000859']:
    print(f'\n=== Filtering field {field} ===', flush=True)
    filter_field(field, overwrite=True)
    print(f'Done with {field}', flush=True)
