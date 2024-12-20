import os
import dill as pickle
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
    if tile.n_bands > 0:
        tile_output_path = tile.store_catalogs(os.path.join(data_path, 'catalog_results'), overwrite=False)
        print(f'Extracted sources from quadrant {field_id} at ({ra_center:.3f}, {dec_center:.3f}). Stored at: {tile_output_path}')
    else:
        print(f'Skipping quadrant extraction {field_id} at ({ra_center:.3f}, {dec_center:.3f}) due to missing bands.')


def process_field(field_id, field_poly, data_path, parallel):
    print(f'Extracting sources from field ID {field_id}...')
    for quadrant in field_poly:
        process_quadrant(quadrant, field_id, data_path, parallel)


def extract_sources():

    # Config
    parallel = False
    data_path = get_data_path()

    # Load the field geometries
    fields = ztffields.Fields()  # TODO: Some filter on fields
    field_info, field_polygons = fields.get_field_vertices([806, 499, 1557, 1558], level='quadrant', steps=2)

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
