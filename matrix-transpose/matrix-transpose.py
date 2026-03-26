import numpy as np

def matrix_transpose(A):
    """
    Return the transpose of matrix A (swap rows and columns).
    """
    # Write code here
    arr = np.array(A)
    rows, cols = arr.shape
    result = np.empty((cols, rows), dtype = arr.dtype)

    for i in range(rows) :
        for j in range(cols):
            result[j, i] = arr[i, j]
    return result
