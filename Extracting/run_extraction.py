import ztffields
import numpy as np

from Tile import Tile


def extract_sources():

    # Config
    parallel = True

    # Load the field geometries
    test = ztffields.Fields()  # TODO: Some filter on fields
    field_info, field_polygons = test.get_field_vertices([1557, 1558], level='quadrant', steps=2)

    # Iterate through the field and quadrants, extracting sources
    for field_poly in field_polygons:
        for quadrant in field_poly:

            # Get the center of the quadrant
            ra_center, dec_center = np.mean(quadrant[:, 0]), np.mean(quadrant[:, 1])

            # Make a tile, run extraction, and store the catalogs
            tile = Tile(
                ra_center,
                dec_center,
                bands=['g', 'r', 'i'],
                data_dir='/Users/adamboesky/Research/long_transients/Data/ztf_data',
                parallel=parallel
            )
            tile.store_catalogs('/Users/adamboesky/Research/long_transients/Data/catalog_results')


if __name__=='__main__':
    extract_sources()
