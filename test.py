import os

from astropy.table import Table

from Extracting.utils import get_data_path
from Source_Analysis.Sources import Source, Sources


cat_num = 1

combined_g_tabs = Table.read(os.path.join(get_data_path(), f'filter_results/combined/{cat_num}_g.ecsv'), format='ascii.ecsv')
combined_r_tabs = Table.read(os.path.join(get_data_path(), f'filter_results/combined/{cat_num}_r.ecsv'), format='ascii.ecsv')
combined_i_tabs = Table.read(os.path.join(get_data_path(), f'filter_results/combined/{cat_num}_i.ecsv'), format='ascii.ecsv')

src = Source(
        ra=12.56254,
        dec=12.88404,
        field_catalogs={
            'g': combined_g_tabs,
            'r': combined_r_tabs,
            'i': combined_i_tabs,
        },
        max_arcsec=3,
        ztf_data_dir='/Users/adamboesky/Research/long_transients/Data/ztf_data',
    )

print(src.data[['ZTF_g_fieldid', 'ZTF_g_ccdid', 'ZTF_g_qid']])
print(src.data[['ZTF_r_fieldid', 'ZTF_r_ccdid', 'ZTF_r_qid']])
print(src.data[['ZTF_i_fieldid', 'ZTF_i_ccdid', 'ZTF_i_qid']])
# for c in src.data.columns:
    # print(c[['fieldid', 'qid']])
    # print(src.data.columns)
