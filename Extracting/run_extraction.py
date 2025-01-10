import os
import sys
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
        try:
            process_quadrant(quadrant, field_id, data_path, parallel)
        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()

            # Navigate to the innermost traceback
            while exc_tb.tb_next:
                next_tb = exc_tb.tb_next
                filename = next_tb.tb_frame.f_code.co_filename
                # Skip if the file is not part of your project (e.g., from site-packages)
                if "site-packages" not in filename and "dist-packages" not in filename:
                    exc_tb = next_tb
                else:
                    break
            fname = exc_tb.tb_frame.f_code.co_filename
            exception_info = f'{fname}:{exc_tb.tb_lineno}\n{exc_type}: {e}'
            print(f'EXCEPTION INFO FOR FIELD {field_id} AT ({np.mean(quadrant[:, 0]):.3f}, {np.mean(quadrant[:, 1]):.3f}):\n{exception_info}')
            if raise_exceptions:
                raise e


def extract_sources():

    # Config
    global raise_exceptions
    raise_exceptions = False
    parallel = False
    data_path = get_data_path()

    # Load the field geometries
    fields = ztffields.Fields()  # TODO: Some filter on fields
    field_info, field_polygons = fields.get_field_vertices([500, 501, 502], level='quadrant', steps=2)

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
                print(f'WARNING: Field processing generated an exception: {exc}')
                raise exc
    # for field_id, field_poly in zip(field_info['fieldid'], field_polygons):
    #     process_field(field_id, field_poly, data_path, parallel)


if __name__=='__main__':
    extract_sources()
