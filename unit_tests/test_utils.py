import numpy as np

from Extracting.utils import nan_nearby, true_nearby


def test_nearby_checks():
    arr = np.zeros((50, 50))
    arr[40, 40] = np.nan
    assert nan_nearby(40, 40, 1, arr) == True
    assert nan_nearby(41, 41, 2, arr) == True
    assert nan_nearby(42, 42, 1, arr) == False
    assert nan_nearby(4, 4, 1, arr) == False

    mask = np.isnan(arr)
    assert true_nearby(40, 40, 1, mask) == True
    assert true_nearby(41, 41, 2, mask) == True
    assert true_nearby(42, 42, 1, mask) == False
    assert true_nearby(4, 4, 1, mask) == False
