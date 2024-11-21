"""Script that will be used to update the forced photometry light curve map csv."""
from Forced_Photo_Map import Forced_Photo_Map


def update_forced_photo_map():
    """Update the forced photometery map."""
    photo_map = Forced_Photo_Map()
    photo_map.add_all_new_light_curves()


if __name__=='__main__':
    update_forced_photo_map()
