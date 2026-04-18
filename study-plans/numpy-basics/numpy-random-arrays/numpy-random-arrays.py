import numpy as np

def generate_random_array(shape, kind, seed):
    """
    Returns: 2D ndarray of float64 random values
    """
    rng = np.random.seed(seed)
    if kind == 'uniform':
        return np.random.rand(*shape)
    elif kind == 'normal':
        return np.random.randn(*shape)
        