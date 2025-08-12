"""NeoWISE light curve queries.

Source: https://irsa.ipac.caltech.edu/docs/notebooks/neowise-source-table-lightcurves.html"""

import multiprocessing  # parallelization

import astroquery.vizier  # fetch the sample of CV targets
import hpgeom  # HEALPix math
import numpy as np  # math
import pandas as pd  # manipulate tabular data
import pyarrow.compute  # construct dataset filters
import pyarrow.dataset  # load and query the NEOWISE dataset
import pyarrow.fs  # interact with the S3 bucket storing the NEOWISE catalog
import pyvo  # TAP service for the Vizier query
from astropy import units as u  # manipulate astropy quantities
from astropy.coordinates import SkyCoord  # manipulate sky coordinates
from matplotlib import pyplot as plt  # plot light curves

# copy-on-write will become the default in pandas 3.0 and is generally more performant
pd.options.mode.copy_on_write = True

# all years => about 11 CPU, 65G RAM, and 50 minutes runtime
YEARS = [f"year{yr}" for yr in range(1, 12)] + ["addendum"]

# To try out a smaller version of the notebook,
# uncomment the next line and choose your own subset of years.
# YEARS = [10]  # one year => about 5 CPU, 20G RAM, and 10 minutes runtime
# sets of columns that we'll need
FLUX_COLUMNS = ["w1flux", "w2flux"]
LIGHTCURVE_COLUMNS = ["mjd"] + FLUX_COLUMNS
COLUMN_SUBSET = ["cntr", "ra", "dec"] + LIGHTCURVE_COLUMNS

# cone-search radius defining which NEOWISE sources are associated with each target object
MATCH_RADIUS = 1 * u.arcsec

# This catalog is so big that even the metadata is big.
# Expect this cell to take about 30 seconds per year.

# This information can be found at https://irsa.ipac.caltech.edu/cloud_access/.
bucket = "nasa-irsa-wise"
base_prefix = "wise/neowiser/catalogs/p1bs_psd/healpix_k5"
metadata_path = (
    lambda yr: f"{bucket}/{base_prefix}/{yr}/neowiser-healpix_k5-{yr}.parquet/_metadata"
)
fs = pyarrow.fs.S3FileSystem(region="us-west-2", anonymous=True)

# list of datasets, one per year
year_datasets = [
    pyarrow.dataset.parquet_dataset(metadata_path(yr), filesystem=fs, partitioning="hive")
    for yr in YEARS
]

# unified dataset, all years
neowise_ds = pyarrow.dataset.dataset(year_datasets)

def get_neowise_lc(ra, dec, radius_arcsec=1, columns=None):
    """
    Get NEOWISE lightcurve for a single coordinate.
    
    Parameters
    ----------
    ra : float
        Right Ascension in degrees
    dec : float
        Declination in degrees
    radius_arcsec : float, optional
        Search radius for cone search in arcseconds. Defaults to MATCH_RADIUS (1 arcsec)
    columns : list, optional
        Columns to include in the output. Defaults to COLUMN_SUBSET
        
    Returns
    -------
    pd.DataFrame
        DataFrame containing the lightcurve data with columns: mjd, w1flux, w2flux, cntr, ra, dec
        Returns empty DataFrame if no data found
    """
    if radius_arcsec is None:
        radius = MATCH_RADIUS
    else:
        radius = radius_arcsec * u.arcsec
    if columns is None:
        columns = COLUMN_SUBSET

    # Convert coordinates to astropy units
    ra = float(ra) * u.deg
    dec = float(dec) * u.deg
    
    # Find HEALPix pixels that contain this coordinate
    healpix_pixels = hpgeom.query_circle(
        a=ra.to_value("deg"),
        b=dec.to_value("deg"),
        radius=radius.to_value("deg"),
        nside=hpgeom.order_to_nside(order=5),
        nest=True,
        inclusive=True,
    )
    
    if len(healpix_pixels) == 0:
        return pd.DataFrame(columns=columns)
    
    # Create a single target DataFrame for this coordinate
    target_df = pd.DataFrame({
        'uid': ['single_target'],
        'RAJ2000': [ra.to_value("deg")],
        'DEJ2000': [dec.to_value("deg")],
        'healpix_k5': [healpix_pixels[0]]  # Use first pixel for simplicity
    })
    
    # Get lightcurve for this single target
    lightcurves = _get_lightcurves_for_targets(target_df, radius, columns)
    
    if len(lightcurves) == 0:
        return pd.DataFrame(columns=columns)
    
    # Return the lightcurve data sorted by MJD
    result = lightcurves[0].sort_values("mjd").reset_index(drop=True)
    
    # Remove the uid column if it's not in the requested columns
    if 'uid' not in columns and 'uid' in result.columns:
        result = result.drop(columns=['uid'])
        
    return result

def _get_lightcurves_for_targets(targets_df, radius, columns):
    """
    Get lightcurves for multiple targets using multiprocessing.
    
    Parameters
    ----------
    targets_df : pd.DataFrame
        DataFrame with targets containing columns: uid, RAJ2000, DEJ2000, healpix_k5
    radius : astropy.Quantity
        Search radius for cone search
    columns : list
        Columns to include in the output
        
    Returns
    -------
    list
        List of DataFrames, one per target
    """
    # Group targets by partition
    targets_groups = targets_df.groupby("healpix_k5")
    
    # Arguments for worker initialization
    init_args = (neowise_ds, columns, radius)
    
    # Use multiprocessing to load lightcurves in parallel
    nworkers = min(8, len(targets_groups))
    chunksize = max(1, len(targets_groups) // nworkers)
    
    with multiprocessing.Pool(nworkers, initializer=_init_worker, initargs=init_args) as pool:
        lightcurves = []
        for lightcurves_df in pool.imap_unordered(
            _load_lightcurves_one_partition, targets_groups, chunksize=chunksize
        ):
            lightcurves.append(lightcurves_df)
    
    return lightcurves

def _load_lightcurves_one_partition(targets_group):
    """Load lightcurves from a single partition.

    Parameters
    ----------
    targets_group : tuple
        Tuple of pixel index and sub-DataFrame (result of DataFrame groupby operation).

    Returns
    -------
    pd.DataFrame
        The lightcurves for targets found in this partition.
    """
    # These global variables will be set when the worker is initialized.
    global _neowise_ds
    global _columns
    global _radius

    # Get row filters that will limit the amount of data loaded from this partition.
    # It is important for these filters to be efficient for the specific use case.
    filters = _construct_dataset_filters(targets_group=targets_group, radius=_radius)

    # Load this slice of the dataset to a pyarrow Table.
    pixel_tbl = _neowise_ds.to_table(columns=_columns, filter=filters)

    # Associate NEOWISE detections with targets to get the light curves.
    lightcurves_df = _cone_search(
        targets_group=targets_group, pixel_tbl=pixel_tbl, radius=_radius
    )

    return lightcurves_df

def _construct_dataset_filters(*, targets_group, radius, scale_factor=100):
    """Construct dataset filters for a box search around all targets in the partition.

    Parameters
    ----------
    targets_group : tuple
        Tuple of pixel index and sub-DataFrame (result of DataFrame groupby operation).
    radius : astropy.Quantity
        The radius used for constructing the RA and Dec filters.
    scale_factor : int (optional)
        Factor by which the radius will be multiplied to ensure that the box encloses
        all relevant detections.

    Returns
    -------
    filters : pyarrow.compute.Expression
        The constructed filters based on the given inputs.
    """
    pixel, locations_df = targets_group

    # Start with a filter for the partition. This is the most important one because
    # it tells the Parquet reader to just skip all the other partitions.
    filters = pyarrow.compute.field("healpix_k5") == pixel

    # Add box search filters. For our CV sample, one box encompassing all targets in
    # the partition should be sufficient. Make a different choice if you use a different
    # sample and find that this loads more data than you want to handle at once.
    buffer_dist = scale_factor * radius.to_value("deg")
    for coord, target_coord in zip(["ra", "dec"], ["RAJ2000", "DEJ2000"]):
        coord_fld = pyarrow.compute.field(coord)

        # Add a filter for coordinate lower limit.
        coord_min = locations_df[target_coord].min()
        filters = filters & (coord_fld > coord_min - buffer_dist)

        # Add a filter for coordinate upper limit.
        coord_max = locations_df[target_coord].max()
        filters = filters & (coord_fld < coord_max + buffer_dist)

    # Add your own additional requirements here, like magnitude limits or quality cuts.
    # See the AllWISE notebook for more filter examples and links to pyarrow documentation.
    # We'll add a filter for sources not affected by contamination or confusion.
    filters = filters & pyarrow.compute.equal(pyarrow.compute.field("cc_flags"), "0000")

    return filters

def _cone_search(*, targets_group, pixel_tbl, radius):
    """Perform a cone search to select NEOWISE detections belonging to each target object.

    Parameters
    ----------
    targets_group : tuple
        Tuple of pixel index and sub-DataFrame (result of DataFrame groupby operation).
    pixel_tbl : pyarrow.Table
        Table of NEOWISE data for a single pixel
    radius : astropy.Quantity
        Cone search radius.

    Returns
    -------
    match_df : pd.DataFrame
        A dataframe with all matched sources.
    """
    _, targets_df = targets_group

    # Cone search using astropy to select NEOWISE detections belonging to each object.
    pixel_skycoords = SkyCoord(ra=pixel_tbl["ra"] * u.deg, dec=pixel_tbl["dec"] * u.deg)
    targets_skycoords = SkyCoord(targets_df["RAJ2000"], targets_df["DEJ2000"], unit=u.deg)
    targets_ilocs, pixel_ilocs, _, _ = pixel_skycoords.search_around_sky(
        targets_skycoords, radius
    )

    # Create a dataframe with all matched source detections.
    match_df = pixel_tbl.take(pixel_ilocs).to_pandas()

    # Add the target IDs by joining with targets_df.
    match_df["targets_ilocs"] = targets_ilocs
    match_df = match_df.set_index("targets_ilocs").join(targets_df.reset_index().uid)

    return match_df

def _init_worker(neowise_ds, columns, radius):
    """Set global variables '_neowise_ds', '_columns', and '_radius'.

    These variables will be the same for every call to '_load_lightcurves_one_partition'
    and will be set once for each worker. It is important to pass 'neowise_ds' this
    way because of its size and the way it will be used. (For the other two, it makes
    little difference whether we use this method or pass them directly as function
    arguments to '_load_lightcurves_one_partition'.)

    Parameters
    ----------
    neowise_ds : pyarrow.dataset.Dataset
        NEOWISE metadata loaded as a PyArrow dataset.
    columns : list
        Columns to include in the output DataFrame of light curves.
    radius : astropy.Quantity
        Cone search radius.
    """
    global _neowise_ds
    _neowise_ds = neowise_ds
    global _columns
    _columns = columns
    global _radius
    _radius = radius

# Legacy functions for backward compatibility
def load_targets_Downes2001(radius=1 * u.arcsec):
    """Load a sample of targets and return a pandas DataFrame.

    References:
    - Downes et al., 2001 ([2001PASP..113..764D](https://ui.adsabs.harvard.edu/abs/2001PASP..113..764D/abstract)).
    - https://cdsarc.cds.unistra.fr/ftp/V/123A/ReadMe

    Parameters
    ----------
    radius : astropy.Quantity (optional)
        Radius for the cone search around each target. This is used to determine which partition(s)
        need to be searched for a given target. Use the same radius here as in the rest of the notebook.

    Returns
    -------
    pandas.DataFrame
        The loaded targets with the following columns:
            - uid: Unique identifier of the target.
            - GCVS: Name in the General Catalogue of Variable Stars if it exists, else the constellation name.
            - RAJ2000: Right Ascension of the target in J2000 coordinates.
            - DEJ2000: Declination of the target in J2000 coordinates.
            - healpix_k5: HEALPix pixel index at order k=5.
    """
    astroquery.vizier.Vizier.ROW_LIMIT = -1
    # https://cdsarc.cds.unistra.fr/vizier/notebook.gml?source=V/123A
    # https://cdsarc.cds.unistra.fr/ftp/V/123A/ReadMe
    CATALOGUE = "V/123A"
    voresource = pyvo.registry.search(ivoid=f"ivo://CDS.VizieR/{CATALOGUE}")[0]
    tap_service = voresource.get_service("tap")

    # Query Vizier and load targets to a dataframe.
    cv_columns = ["uid", "GCVS", "RAJ2000", "DEJ2000"]
    cvs_records = tap_service.run_sync(
        f'SELECT {",".join(cv_columns)} from "{CATALOGUE}/cv"'
    )
    cvs_df = cvs_records.to_table().to_pandas()

    # Add a new column containing a list of all order k HEALPix pixels that overlap with
    # the CV's position plus search radius.
    cvs_df["healpix_k5"] = [
        hpgeom.query_circle(
            a=cv.RAJ2000,
            b=cv.DEJ2000,
            radius=radius.to_value("deg"),
            nside=hpgeom.order_to_nside(order=5),
            nest=True,
            inclusive=True,
        )
        for cv in cvs_df.itertuples()
    ]
    # Explode the lists of pixels so the dataframe has one row per target per pixel.
    # Targets near a pixel boundary will now have more than one row.
    # Later, we'll search each pixel separately for NEOWISE detections and then
    # concatenate the matches for each target to produce complete light curves.
    cvs_df = cvs_df.explode("healpix_k5", ignore_index=True)

    return cvs_df

def load_lightcurves_one_partition(targets_group):
    """Legacy function - use _load_lightcurves_one_partition instead."""
    return _load_lightcurves_one_partition(targets_group)

def init_worker(neowise_ds, columns, radius):
    """Legacy function - use _init_worker instead."""
    return _init_worker(neowise_ds, columns, radius)

# Example usage and demonstration code
if __name__ == "__main__":
    # Example: Get lightcurve for a specific coordinate
    ra_example = 150.0  # degrees
    dec_example = 2.0   # degrees
    
    print(f"Getting NEOWISE lightcurve for RA={ra_example}, Dec={dec_example}")
    lc = get_neowise_lc(ra_example, dec_example)
    
    if len(lc) > 0:
        print(f"Found {len(lc)} data points")
        print(lc.head())
    else:
        print("No data found for this coordinate")

    plt.plot(lc.mjd, lc.w1flux, ".")
