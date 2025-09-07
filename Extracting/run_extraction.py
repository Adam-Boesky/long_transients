import os
import sys
import ztffields
import numpy as np
import traceback

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


def add_to_bad_quads(quad_dirname: str):
    """Add a quadrant to the bad quadrants list."""
    bad_quads_fpath = os.path.join(get_data_path(), 'bad_quads.npy')
    if os.path.exists(bad_quads_fpath):
        bad_quads = np.load(bad_quads_fpath)
        bad_quads = np.append(bad_quads, quad_dirname)
        os.remove(bad_quads_fpath)
    else:
        bad_quads = np.array([quad_dirname])
    
    # Save the bad quadrants
    with open(bad_quads_fpath, 'wb') as f:
        print('saving bad quads, ', bad_quads)
        np.save(f, bad_quads)


def process_quadrant(fieldid: int, ccdid: int, qid: int, bands: Iterable[str]):

    try:
        # Get the quadrant directory name (ffff_cc_qq)
        quad_dirname = f'{str(fieldid).zfill(6)}_{str(ccdid).zfill(2)}_{qid}'

        # quads with no pts
        bad_quads_fpath = os.path.join(get_data_path(), 'bad_quads.npy')
        if os.path.exists(bad_quads_fpath):
            print('loading bad quads...')
            bad_quads = np.load(bad_quads_fpath)
            if quad_dirname in bad_quads:
                print(f'{quad_dirname} is in bad_quads. Skipping!')
                return

        # Check if we have already extracted the quadrant before doing anything
        data_path = get_data_path()
        if os.path.exists(
            os.path.join(
                data_path,
                'catalog_results',
                quad_dirname
            )
        ):
            print(f'{quad_dirname} already exists. Skipping!')
            return

        print(f'Extracting sources from quadrant with field_id={fieldid}, ccdid={ccdid}, qid={qid}...')

        # Get data
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

    except Exception as e:
        if isinstance(e, ValueError) and 'No points given' in str(e):
            print('No points given. Adding to bad_quads.npy')
            add_to_bad_quads(quad_dirname)

        elif isinstance(e, ValueError) and 'The truth value of an array with more than one element is ambiguous' in str(e):
            print('The truth value of an array with more than one element is ambiguous. Adding to bad_quads.npy')
            add_to_bad_quads(quad_dirname)

        elif isinstance(e, Exception) and 'The limit of 300000 active object pixels over the detection threshold' in str(e):
            print('The limit of 300000 active object pixels over the detection threshold was reached. Adding to bad_quads.npy')
            add_to_bad_quads(quad_dirname)

        raise e


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
                print(f'WARNING: Field processing generated an exception: {exc.__class__.__name__}: {exc}')
                print(f'Exception type: {type(exc).__name__}')
                print(f'Exception message: {str(exc)}')
                print('Traceback:')
                traceback.print_exc()

                if raise_exceptions:
                    raise exc


def extract_sources():

    # Config
    global raise_exceptions
    raise_exceptions = False
    data_path = get_data_path()

    # Load the field geometries
    print('Loading fields!')
    # Load array
    bands_imaged = {}
    for band in ('g', 'r', 'i'):
        bands_imaged[band] = np.load(os.path.join(data_path, f'{band}_imaged_fields.npy'))
    fields_imaged_all_bands = np.intersect1d(ar1=bands_imaged['g'], ar2=bands_imaged['r'])
    fields_imaged_all_bands = np.intersect1d(ar1=fields_imaged_all_bands, ar2=bands_imaged['i'])

    # THIS IS THE FINAL RUN!!!
    for fid in fields_imaged_all_bands[100:]:
        process_field(fid)

if __name__=='__main__':
    extract_sources()
