import ztffields
import numpy as np

from Extracting.Tile import Tile


def extract_sources():

    # Load the field geometries
    test = ztffields.Fields()  # TODO: Some filter on fields
    field_info, field_polygons = test.get_field_vertices([1557, 1558], level='quadrant', steps=2)

    # Iterate through the field and quadrants, extracting sources
    for field_poly in field_polygons:
        for quadrant in field_poly:

            # Get the center of the quadrant
            ra_center, dec_center = np.mean(quadrant[:, 0]), np.mean(quadrant[:, 1])

            # Make a tile and run extraction
            tile = Tile(
                ra_center,
                dec_center,
                bands=['g', 'r', 'i'],
                data_dir='/Users/adamboesky/Research/long_transients/Data/ztf_data',
            )
            data_dicts = tile.data_dicts  # TODO: Figure out how to store nicely
            print(data_dicts)


if __name__=='__main__':
    extract_sources()
