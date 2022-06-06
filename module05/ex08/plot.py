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

def loss_(y, y_hat):
    if type(y_hat) != np.ndarray or type(y) != np.ndarray:
        return None
    if len(y) == 0 or len(y_hat) == 0:
        return None
    if y.shape != y_hat.shape:
        return None
    result = float(np.dot((y_hat - y), (y_hat - y)) / (2 * y.size))
    print(result)
    return result

def plot_with_loss(x, y, theta):
    """Plot the data and prediction line from three non-empty numpy.array.
    Args:
    x: has to be an numpy.array, a vector of dimension m * 1.
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
    xpoints1 = [x[0], x[x.size - 1]]
    ypoints1 = [predict_(x, theta)[0], predict_(x, theta)[x.size - 1]]
    plt.plot(xpoints1, ypoints1)
    y_hat = predict_(x, theta)
    print(loss_(y, y_hat))
    for i in range(y_hat.size):
        plt.plot([x[i], x[i]], [y[i], y_hat[i]], linestyle='dotted')
    plt.title(str("Cost : " + str(loss_(y, y_hat))))
    plt.show()
