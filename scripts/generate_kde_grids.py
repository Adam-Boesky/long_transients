"""
Generate KDE grid files (X.npy, Y.npy, Z_{band}.npy) from pre-fit KDE pkl files.

Loads the per-band KDEs from Data/{band}_kde.pkl, evaluates them on the same
mesh grid used in calculate_5sigma.ipynb, and saves the arrays to Data/.
"""

import os
import sys
import pickle

import numpy as np

sys.path.append('/Users/adamboesky/Research/long_transients')
from Extracting.utils import get_data_path

BANDS = ['g', 'r', 'i']

# Grid matching calculate_5sigma.ipynb Cell 15
xgrid = np.linspace(11, 25, 100)   # PanSTARRS mag range
ygrid = np.linspace(-5, 5, 100)    # ZTF - PanSTARRS dmag range
X, Y = np.meshgrid(xgrid, ygrid)
positions = np.vstack([X.ravel(), Y.ravel()])

data_path = get_data_path()

# Save the shared grid axes once
x_out = os.path.join(data_path, 'X.npy')
y_out = os.path.join(data_path, 'Y.npy')
np.save(x_out, X)
np.save(y_out, Y)
print(f'Saved {x_out}')
print(f'Saved {y_out}')

# Evaluate and save per-band Z grids
for band in BANDS:
    kde_path = os.path.join(data_path, f'{band}_kde.pkl')
    with open(kde_path, 'rb') as f:
        kde = pickle.load(f)

    print(f'Evaluating KDE for band {band}...')
    Z = np.reshape(kde(positions).T, X.shape)

    z_out = os.path.join(data_path, f'Z_{band}.npy')
    np.save(z_out, Z)
    print(f'Saved {z_out}')

print('Done.')
