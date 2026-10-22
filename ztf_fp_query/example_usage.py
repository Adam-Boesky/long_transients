"""Example usage of the ztf_fp_query module.

Workflow:
  1. submit()  — send coordinates to the IPAC forced-photometry service
  2. sync()    — pull completed results to local storage (run after jobs finish)
  3. ForcedPhotometry — load and work with the data
"""

import os
import sys
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ztf_fp_query.submit import submit, query_recent_jobs, query_pending_jobs, recently_queried, currently_pending
from ztf_fp_query.sync import sync
from ztf_fp_query.forced_photometry import ForcedPhotometry


# -----------------------------------------------------------------------
# 1. Submit a batch request
# -----------------------------------------------------------------------

ras  = [210.08983, 206.42724]
decs = [ -6.88676,  -8.13603]

# Check which positions actually need submitting
already_done    = recently_queried(ras, decs)
still_pending   = currently_pending(ras, decs)
needs_submit    = ~already_done & ~still_pending

if needs_submit.any():
    submit([r for r, s in zip(ras, needs_submit) if s],
           [d for d, s in zip(decs, needs_submit) if s])
    print('Submitted. The service will email you when jobs complete.')
else:
    print('All positions already submitted or pending.')


# -----------------------------------------------------------------------
# 2. Check job status
# -----------------------------------------------------------------------

print('\n--- Pending jobs ---')
print(query_pending_jobs())

print('\n--- Recent jobs ---')
print(query_recent_jobs())


# -----------------------------------------------------------------------
# 3. Sync completed results to local storage
# -----------------------------------------------------------------------

downloaded = sync()
print(f'\nDownloaded {len(downloaded)} new file(s).')


# -----------------------------------------------------------------------
# 4. Load a ForcedPhotometry object
# -----------------------------------------------------------------------

# By sky position (looks up the file via the local index)
fp = ForcedPhotometry(ra=210.08983, dec=-6.88676)
print(fp)  # ForcedPhotometry(ra=..., dec=..., n_epochs=..., bands=[...])

# Or directly from a file path
fp = ForcedPhotometry.from_file(
    '/Users/adamboesky/Research/long_transients/Data/ztf_forced_photometry/'
    'batchfp_req0004838848_lc.txt'
)


# -----------------------------------------------------------------------
# 5. Access the data
# -----------------------------------------------------------------------

# Full DataFrame — all ZFPS columns plus derived: snr, upperlim, mag, magerr
print(fp.df.columns.tolist())
print(fp.df.head())

# Which bands are present?
print(fp.bands)  # e.g. ['ZTF_g', 'ZTF_r', 'ZTF_i']

# Single-band slice
g_lc = fp.get_band('ZTF_g')
print(f'g-band epochs: {len(g_lc)}')

# Detections vs upper limits
detections  = fp.df[fp.df['upperlim'] == 0]
upper_limits = fp.df[fp.df['upperlim'] == 1]
print(f'Detections: {len(detections)}, Upper limits: {len(upper_limits)}')

# Quality filtering — done directly on the DataFrame
good = fp.df[(fp.df['procstatus'] == 0) & (fp.df['diffimgstatus'] == 1)]
print(f'Good-quality epochs: {len(good)}')

# High-SNR g-band detections
bright_g = fp.get_band('ZTF_g')
bright_g = bright_g[(bright_g['upperlim'] == 0) & (bright_g['snr'] > 10)]
print(f'High-SNR g-band epochs: {len(bright_g)}')


# -----------------------------------------------------------------------
# 6. Stacking
# -----------------------------------------------------------------------

# Stack all epochs — useful for placing a deep limit on a quiescent source
print(fp.stack())

# Stack over a pre-event baseline window
jd_min = fp.df['jd'].min()
print(fp.stack(jd_min=jd_min, jd_max=jd_min + 200, bands=['ZTF_g', 'ZTF_r']))


# -----------------------------------------------------------------------
# 7. Plot
# -----------------------------------------------------------------------

# Default: magnitude vs time, all bands
fig, ax = plt.subplots(figsize=(12, 5))
fp.plot(ax=ax)
ax.set_title('ZTF forced photometry')
plt.tight_layout()
plt.show()

# Flux, g and r only, custom time offset
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

fp.plot(ax=axes[0], y_units='mag', bands=['ZTF_g', 'ZTF_r'])
axes[0].set_title('Magnitude')

fp.plot(ax=axes[1], y_units='flux', bands=['ZTF_g', 'ZTF_r'])
axes[1].set_title('Flux')

plt.tight_layout()
plt.show()
