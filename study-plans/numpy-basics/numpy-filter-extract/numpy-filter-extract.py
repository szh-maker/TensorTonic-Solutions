import numpy as np

def filter_and_extract(data, row_start, row_stop, threshold):
    """
    Returns: 1D ndarray of float64
    """
    arr = np.asarray(data, dtype=np.float64)
    block = arr[row_start : row_stop]
    result = block[block > threshold]
    return result.astype(np.float64)