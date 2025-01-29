import os
import sys
import ztffields
import numpy as np

from typing import Iterable
from concurrent.futures import ProcessPoolExecutor, as_completed

try:
    from Tile import Tile
    from Catalogs import ztf_image_exists, get_ztf_metadata_from_metadata
    from utils import get_data_path
except ModuleNotFoundError:
    from .Tile import Tile
    from .Catalogs import ztf_image_exists, get_ztf_metadata_from_metadata
    from .utils import get_data_path


def process_quadrant(fieldid: int, ccdid: int, qid: int, bands: Iterable[str]):
    print(f'Extracting sources from quadrant with field_id={fieldid}, ccdid={ccdid}, qid={qid}...')

    # Get data
    data_path = get_data_path()
    ztf_metadata={
        'fieldid': fieldid,
        'ccdid': ccdid,
        'qid': qid,
    }

    # Run extraction and store
    tile = Tile(
        ztf_metadata=ztf_metadata,
        bands=bands,
        data_dir=os.path.join(data_path, 'ztf_data'),
    )
    tile_output_path = tile.store_catalogs(os.path.join(data_path, 'catalog_results'), overwrite=False)
    print(f'Extracted sources from field with metadata {ztf_metadata}. Stored at: {tile_output_path}')


def process_field(field_id: int):
    # Get the field dataframe
    field_df = get_ztf_metadata_from_metadata(ztf_metadata={'fieldid': field_id})

    # Iterate through the field, extracting sources with quadrants parallelized
    with ProcessPoolExecutor(max_workers=8) as executor:
        futures = [
            executor.submit(
                process_quadrant,
                fieldid,
                ccdid,
                qid,
                [c[1] for c in df['filtercode']]
            ) for (fieldid, ccdid, qid), df in field_df.groupby(['field', 'ccdid', 'qid'])
        ]

        for future in as_completed(futures):
            try:
                # Call result() to propagate any exceptions that occurred
                future.result()
            except Exception as exc:
                print(f'WARNING: Field processing generated an exception: {exc}')
                if raise_exceptions:
                    raise exc


def extract_sources():

    # Config
    global raise_exceptions
    raise_exceptions = False
    data_path = get_data_path()

    # Load the field geometries
    print('Loading fields!')
    imaged_fields = np.load(os.path.join(data_path, 'imaged_fields.npy'))

    # Extract field
    for fid in imaged_fields:
        process_field(fid)


if __name__=='__main__':
    extract_sources()
