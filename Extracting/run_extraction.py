import multiprocessing
import os
import sys
import pickle
import ztffields
import numpy as np
import traceback

from typing import Iterable
from concurrent.futures import ProcessPoolExecutor, as_completed

try:
    from Tile import Tile
    from Catalogs import ztf_image_exists, get_ztf_metadata_from_metadata
    from utils import get_data_path, init_worker_lock
except ModuleNotFoundError:
    from .Tile import Tile
    from .Catalogs import ztf_image_exists, get_ztf_metadata_from_metadata
    from .utils import get_data_path, init_worker_lock


def add_to_bad_quads(quad_dirname: str):
    """Add a quadrant to the bad quadrants list."""
    print(f'Adding {quad_dirname} to bad_quads.npy')
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


def get_incomplete_quadrant_dirs(bands: Iterable[str] = ('g', 'r', 'i')) -> dict:
    """Crawl catalog_results and return directories that are missing expected extraction files.

    A complete quadrant directory contains:
      - PSTARR.hdf5
      - For each band present: ZTF_{band}.hdf5, nan_masks/ZTF_{band}_nan_mask.npy,
        WCSs/ZTF_{band}_wcs.pkl, EPSFs/ZTF_{band}_EPSF.npy

    'Bands present' is inferred from whichever ZTF_{band}.hdf5 files exist. If none
    exist for any band, the whole directory is flagged as empty.

    Returns:
        Dict mapping quad_dirname -> list of missing file paths (relative to the quad dir).
    """
    results_dir = os.path.join(get_data_path(), 'catalog_results')
    incomplete = {}

    results_dirs = os.listdir(results_dir)
    for i, dirname in enumerate(sorted(results_dirs)):
        if i % 100 == 0:
            print(f'Checking completeness of {i} / {len(results_dirs)} directories...')
        dirpath = os.path.join(results_dir, dirname)
        if not os.path.isdir(dirpath) or dirname == 'field_results':
            continue

        # List each subdirectory once and check membership against sets — avoids
        # one network stat() call per file on NFS/Lustre filesystems.
        try:
            root_files = set(os.listdir(dirpath))
        except OSError:
            incomplete[dirname] = ['(could not list directory contents)']
            continue

        missing = []

        if 'PSTARR.hdf5' not in root_files:
            missing.append('PSTARR.hdf5')

        for band in bands:
            if f'ZTF_{band}.hdf5' not in root_files:
                missing.append(f'ZTF_{band}.hdf5')

        if missing:
            incomplete[dirname] = missing

    return incomplete


def process_quadrant(fieldid: int, ccdid: int, qid: int, bands: Iterable[str]):

    try:
        # Get the quadrant directory name (ffff_cc_qq)
        quad_dirname = f'{str(fieldid).zfill(6)}_{str(ccdid).zfill(2)}_{qid}'

        # quads with no pts
        bad_quads_fpath = os.path.join(get_data_path(), 'bad_quads.npy')
        if os.path.exists(bad_quads_fpath):
            bad_quads = np.load(bad_quads_fpath)
            if quad_dirname in bad_quads:
                print(f'{quad_dirname} is in bad_quads. Skipping!')
                return

        # Check if we have already extracted the quadrant before doing anything
        data_path = get_data_path()
        bands_not_extracted = []
        for band in bands:
            if not os.path.exists(
                os.path.join(
                    data_path,
                    'catalog_results',
                    quad_dirname,
                    f'ZTF_{band}.hdf5'
                )
            ):
                bands_not_extracted.append(band)
        if len(bands_not_extracted) == 0:
            print(f'{quad_dirname} in {bands} already exists. Skipping!')
            return

        # Only extract the bands that are not already extracted
        bands = bands_not_extracted

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
        tile_output_path = tile.store_catalogs(os.path.join(data_path, 'catalog_results'), overwrite=True)
        print(f'Extracted sources from field with metadata {ztf_metadata}. Stored at: {tile_output_path}')

    except Exception as e:
        error_msg = str(e)
        print(f"DEBUG: Caught exception: {type(e).__name__}: {error_msg}")
        
        if 'No points given' in error_msg:
            print('No points given.')
            add_to_bad_quads(quad_dirname)

        # elif 'The truth value of an array with more than one element is ambiguous' in error_msg:
        #     print('The truth value of an array with more than one element is ambiguous.')
        #     add_to_bad_quads(quad_dirname)

        # elif 'The limit of 300000 active object pixels over the detection threshold' in error_msg:
        #     print('The limit of 300000 active object pixels over the detection threshold was reached.')
        #     add_to_bad_quads(quad_dirname)

        # For any other exceptions, re-raise them
        print(f"DEBUG: Re-raising unhandled exception: {type(e).__name__}: {error_msg}")
        traceback.print_exc()

        raise e


def process_field(field_id: int):
    # Get the field dataframe
    field_df = get_ztf_metadata_from_metadata(ztf_metadata={'fieldid': field_id})

    # Shared lock so that worker processes serialise CasJobs API calls
    # (the service forbids concurrent DataReaders on the same account).
    casjobs_lock = multiprocessing.Lock()

    # Iterate through the field, extracting sources with quadrants parallelized
    with ProcessPoolExecutor(max_workers=8, initializer=init_worker_lock, initargs=(casjobs_lock,)) as executor:
        futures = [
            executor.submit(
                process_quadrant,
                fieldid,
                ccdid,
                qid,
                list(dict.fromkeys(c[1] for c in df['filtercode']))
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


def process_missed_quadrants(quads_to_reextract: dict):
    """Re-extract quadrants that are missing HDF5 files.

    For each incomplete quadrant, queries ZTF metadata to confirm which missing
    bands actually have images available, then runs extraction on those bands.

    Parameters:
        quads_to_reextract: dict mapping quad_dirname -> list of missing band characters
                            (e.g. ['g', 'r']), derived from get_incomplete_quadrant_dirs().
    """
    data_path = get_data_path()

    # Load persistent metadata cache (keyed by (fieldid, ccdid, qid))
    metadata_cache_path = os.path.join(data_path, 'ztf_metadata_cache.pkl')
    if os.path.exists(metadata_cache_path):
        with open(metadata_cache_path, 'rb') as f:
            metadata_cache = pickle.load(f)
    else:
        metadata_cache = {}

    # Build a list of (fieldid, ccdid, qid, bands_to_extract) for quadrants that
    # have at least one missing band with an available ZTF image.
    quadrants_to_run = []
    for dirname, missing_bands in quads_to_reextract.items():
        fieldid, ccdid, qid = dirname.split('_')
        fieldid, ccdid, qid = int(fieldid), int(ccdid), int(qid)

        if not missing_bands:
            continue

        # Query ZTF metadata once per quadrant to see which bands have images,
        # using the on-disk cache to avoid redundant network calls across sessions.
        cache_key = (fieldid, ccdid, qid)
        if cache_key not in metadata_cache:
            print(f'Querying metadata for {dirname}...')
            metadata_cache[cache_key] = get_ztf_metadata_from_metadata(
                ztf_metadata={'fieldid': fieldid, 'ccdid': ccdid, 'qid': qid},
                verbose=0,
            )
            with open(metadata_cache_path, 'wb') as f:
                pickle.dump(metadata_cache, f)
        else:
            print(f'Using cached metadata for {dirname}.')
        metadata = metadata_cache[cache_key]
        available_bands = [fc[1] for fc in metadata['filtercode'].unique()]
        bands_to_extract = [b for b in missing_bands if b in available_bands]

        if not bands_to_extract:
            print(f'{dirname}: no ZTF images available for missing bands {missing_bands}. Skipping.')
            continue

        print(f'{dirname}: will re-extract bands {bands_to_extract}.')
        quadrants_to_run.append((fieldid, ccdid, qid, bands_to_extract))

    print(f'Re-extracting {len(quadrants_to_run)} quadrants...')
    casjobs_lock = multiprocessing.Lock()
    with ProcessPoolExecutor(max_workers=8, initializer=init_worker_lock, initargs=(casjobs_lock,)) as executor:
        futures = [
            executor.submit(process_quadrant, fieldid, ccdid, qid, bands)
            for fieldid, ccdid, qid, bands in quadrants_to_run
        ]
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as exc:
                print(f'WARNING: Re-extraction generated an exception: {exc.__class__.__name__}: {exc}')
                traceback.print_exc()


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

    # # Gemini logic
    # north_fields = ztffields.get_fieldid(grid='main', dec_range=[10, 30], ra_range=[120, 170])
    # south_fields = ztffields.get_fieldid(grid='main', dec_range=[-30, -10], ra_range=[130, 180])
    # gemini_fields = np.concatenate([north_fields, south_fields])
    # fields_imaged_all_bands = np.intersect1d(ar1=gemini_fields, ar2=fields_imaged_all_bands)

    # # Get incomplete quadrants that belong to imaged fields
    # print('Checking for incomplete quadrant directories...')
    # incomplete_quads = get_incomplete_quadrant_dirs()
    # quads_to_reextract = {
    #     dirname: [f[4] for f in missing_files if f.startswith('ZTF_') and f.endswith('.hdf5')]
    #     for dirname, missing_files in incomplete_quads.items()
    #     if int(dirname[:6]) in fields_imaged_all_bands
    #     and any(f.startswith('ZTF_') and f.endswith('.hdf5') for f in missing_files)
    # }
    # print(f'Found {len(quads_to_reextract)} incomplete quadrants to re-extract.')

    # # fields_imaged_all_bands = ['000315', '000316', '000573']

    # process_missed_quadrants(quads_to_reextract)
    # # THIS IS THE FINAL RUN!!!
    for fid in fields_imaged_all_bands:
        process_field(fid)

if __name__=='__main__':
    extract_sources()
