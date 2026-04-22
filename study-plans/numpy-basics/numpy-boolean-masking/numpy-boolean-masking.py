import numpy as np

def row_summary(data, threshold):
    """Returns: np.ndarray of shape (3, m, n), stacked element mask, any-filtered, all-filtered"""
    data = np.asarray(data, dtype=np.float64)
    cond = data > threshold
    mask = cond.astype(np.float64)
    any_filtered = np.where(np.any(cond, axis=1, keepdims=True), data, 0.0)
    all_filtered = np.where(np.all(cond, axis=1, keepdims=True), data, 0.0)
    return np.stack([mask, any_filtered, all_filtered], axis=0)