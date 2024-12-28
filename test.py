import sys

sys.path.append('/Users/adamboesky/Research/long_transients')

from Extracting.Tile import Tile

tile = Tile(96.36058280595702, 12.249594810753777)
if tile.n_bands > 0:
    tile_output_path = tile.store_catalogs('/Users/adamboesky/Research/long_transients/Data/test_extraction', overwrite=True)
tile_output_path