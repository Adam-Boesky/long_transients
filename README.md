# long_transients

## Workflow Overview
### 1. Extract
This step extracts sources from ZTF deep stack reference images and the Pan-STARRS source catalog, and stores them.
- Run `run_extraction.py` with the option of the bash script `run_extraction.sh`.
- Ensure that directories populate into the `path/to/Data/catalog_results/field_results` directory with names like `<field_id>_<ccd_id>_<q_id>` (e.g. `000245_15_4`) that contain the following files:
    - EPSFs/
        - ZTF_g_EPSF.npy
        - ZTF_r_EPSF.npy
        - ZTF_i_EPSF.npy
    - nan_masks/
        - ZTF_g_nan_mask.npy
        - ZTF_r_nan_mask.npy
        - ZTF_i_nan_mask.npy 
    - WCSs/
        - ZTF_g_wcs.pkl
        - ZTF_r_wcs.pkl
        - ZTF_i_wcs.pkl
    - PSTARR.ecsv
    - ZTF_g.ecsv  
    - ZTF_r.ecsv
    - ZTF_i.ecsv


### 2. Match sources between ZTF and Pan-STARRS
This step will associate the sources stored in the ZTF and Pan-STARRS `.ecsv` files that were stored in step 1.
- Run `cross_match.py -aq -mf` with the option of the bash script `cross_match.sh`.
    - If you don't want to associate quadrants and merge fields, consider removing the kwargs.
- Ensure that files populate into the `path/to/Data/catalog_results/field_results` directory with names like `<field_id>_<band>.ecsv`. Note that you'll also find files like `<band>_associated.ecsv` in the `<field_id>_<ccd_id>_<q_id>` directories created in step 1.
It is important to realize that sources may have non-detections in Pan-STARRS or ZTF!

### 3. Filter sources
This step takes the sources and passes them through filtration pipelines. You'll just need to look at the `filter_fields.py` file.

### 4. Make (and sort) analysis pages
Make analysis pages for each source by running `store_src_plots.py` which does something like:
```
import matplotlib.pyplot as plt
from Source_Analysis.Sources import Sources

coords = Sources.from_file('path/to/file.ecsv')
for c in coords:
    c.plot_everything()
    plt.show()
```
