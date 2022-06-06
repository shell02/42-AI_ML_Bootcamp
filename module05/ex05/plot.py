import numpy as np
import matplotlib.pyplot as plt

def add_intercept(x):
    if type(x) != np.ndarray or len(x) == 0:
        return None
    if x.ndim == 1:
        x = x.reshape(x.size, 1)
    return np.insert(x, 0, 1, axis=1)

def predict_(x, theta):
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

def plot(x, y, theta):
    """Plot the data and prediction line from three non-empty numpy.array.
    Args:
    x: has to be an numpy.array, a vector of dimension m * n.
    y: has to be an numpy.array, a vector of dimension m * 1.
    theta: has to be an numpy.array, a vector of dimension 2 * 1.
    Returns:
    Nothing.
    Raises:
    This function should not raise any Exceptions.
    """
    if type(x) != np.ndarray or type(y) != np.ndarray or type(theta) != np.ndarray:
        return
    if len(x) == 0 or len(y) == 0 or len(theta) == 0:
        return
    if x.ndim == 1:
        x = x.reshape(x.size, 1)
    if theta.ndim == 1:
        theta = theta.reshape(theta.size, 1)
    if theta.shape[1] != 1 or theta.shape[0] != 2:
        return None
    plt.scatter(x, y)
    xpoints = [x[0], x[x.size - 1]]
    ypoints = [predict_(x, theta)[0], predict_(x, theta)[x.size - 1]]
    plt.plot(xpoints, ypoints)
    plt.show()

