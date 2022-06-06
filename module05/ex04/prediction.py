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

def predict_(x, theta):
    """Computes the vector of prediction y_hat from two non-empty numpy.array.
    Args:
    x: has to be an numpy.array, a vector of dimension m * n.
    theta: has to be an numpy.array, a vector of dimension 2 * 1.
    Returns:
    y_hat as a numpy.array, a vector of dimension m * 1.
    None if x and/or theta are not numpy.array.
    None if x or theta are empty numpy.array.
    None if x or theta dimensions are not appropriate.
    Raises:
    This function should not raise any Exceptions.
    """
    if type(x) != np.ndarray or type(theta) != np.ndarray:
        return None
    if x.ndim == 1:
        x = x.reshape(x.size, 1)
    if theta.ndim == 1:
        theta = theta.reshape(theta.size, 1)
    if theta.shape[1] != 1 or theta.shape[0] != 2:
        return None
    x = add_intercept(x)
    return np.dot(x, theta).flatten()
