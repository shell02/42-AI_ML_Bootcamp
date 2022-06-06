import numpy as np

def simple_predict(x, theta):
    """Computes the vector of prediction y_hat from two non-empty numpy.ndarray.
    Args:
    x: has to be an numpy.ndarray, a vector of dimension m * 1.
    theta: has to be an numpy.ndarray, a vector of dimension 2 * 1.
    Returns:
    y_hat as a numpy.ndarray, a vector of dimension m * 1.
    None if x or theta are empty numpy.ndarray.
    None if x or theta dimensions are not appropriate.
    Raises:
    This function should not raise any Exception.
    """
    if type(x) != np.ndarray or type(theta) != np.ndarray:
        return None
    if len(x) == 0 or len(theta) == 0:
        return None
    if x.ndim != 1 or theta.ndim != 1:
        return None
    if x.size <= 2 or theta.size != 2:
        return None
    y_hat = np.zeros(x.size)
    for col in range(x.size):
        y_hat[col] = theta[0] + theta[1] * x[col]
    return y_hat
