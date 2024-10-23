import os
import pickle
import ztffields
import numpy as np

from concurrent.futures import ProcessPoolExecutor, as_completed

try:
    from Tile import Tile
    from utils import get_data_path
except ModuleNotFoundError:
    from .Tile import Tile
    from .utils import get_data_path


def process_quadrant(quadrant, field_id, data_path, parallel):
    # Get the center of the quadrant
    ra_center, dec_center = np.mean(quadrant[:, 0]), np.mean(quadrant[:, 1])

    # Make a tile, run extraction, and store the catalogs
    tile = Tile(
        ra_center,
        dec_center,
        bands=['g', 'r', 'i'],
        data_dir=os.path.join(data_path, 'ztf_data'),
        parallel=parallel
    )
    tile_output_path = tile.store_catalogs(os.path.join(data_path, 'catalog_results'), overwrite=True)

    # For testing, store the tile objects as well (NOTE: DELTE THIS IN THE FUTURE)
    with open(os.path.join(tile_output_path, 'tile.pkl'), 'wb') as f:
        pickle.dump(tile, f)


def process_field(field_id, field_poly, data_path, parallel):
    print(f'Extracting sources from field ID {field_id}...')
    for quadrant in field_poly:
        process_quadrant(quadrant, field_id, data_path, parallel)


def extract_sources():

    # Config
    parallel = False
    data_path = get_data_path()

    # Load the field geometries
    test = ztffields.Fields()  # TODO: Some filter on fields
    field_info, field_polygons = test.get_field_vertices([1557, 1558], level='quadrant', steps=2)

    # Iterate through the field and quadrants, extracting sources
    with ProcessPoolExecutor(max_workers=8) as executor:
        futures = [
            executor.submit(process_field, field_id, field_poly, data_path, parallel)
            for field_id, field_poly in zip(field_info['fieldid'], field_polygons)
        ]
        for future in as_completed(futures):
            try:
                # Call result() to propagate any exceptions that occurred
                future.result()
            except Exception as exc:
                print(f'Field processing generated an exception: {exc}')
                raise exc
    # for field_id, field_poly in zip(field_info['fieldid'], field_polygons):
    #     process_field(field_id, field_poly, data_path, parallel)


if __name__=='__main__':
    extract_sources()
