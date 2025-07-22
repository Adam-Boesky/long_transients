import os
import json

from utils import get_n_quadrants_merged_all_fields, get_data_path


def store_extraction_info():
    """Store the extraction info for each candidate."""
    n_quadrants_merged = get_n_quadrants_merged_all_fields()
    data_path = get_data_path()

    # Prepare the output directory and file path
    output_file = os.path.join(data_path, 'catalog_results/extraction_info.json')

    # Store n_quadrants_merged in the JSON file
    with open(output_file, 'w') as f:
        json.dump(n_quadrants_merged, f, indent=4)


if __name__ == '__main__':
    store_extraction_info()
