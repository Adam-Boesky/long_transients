"""Re-run one field with the current code into a scratch dir, to test reproducibility."""
import sys
sys.path.insert(0, '/n/home04/aboesky/berger/long_transients')
import Source_Analysis.filter_fields as ff

FIELD = sys.argv[1] if len(sys.argv) > 1 else '000276'
ff.FILTER_RESULT_DIR = 'determinism_test'
print(f'refiltering {FIELD} into {ff.FILTER_RESULT_DIR}', flush=True)
ff.filter_field(FIELD, overwrite=True, store_pre_gaia=False)
print('done', flush=True)
