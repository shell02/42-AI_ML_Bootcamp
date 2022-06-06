import numpy as np

def add_intercept(x):
    """Adds a column of 1’s to the non-empty numpy.array x.
    Args:
    x: has to be a numpy.array of dimension m * n.
    Returns:
    X, a numpy.array of dimension m * (n + 1).
    None if x is not a numpy.array.
    None if x is an empty numpy.array.
    Raises:
    This function should not raise any Exception.
    """
    if type(x) != np.ndarray or len(x) == 0:
        return None
    if x.ndim == 1:
        x = x.reshape(x.size, 1)
    return np.insert(x, 0, 1, axis=1)
