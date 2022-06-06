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
    theta: has to be an numpy.array, a vector of dimension n * 1.
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
    x = add_intercept(x)
    return np.dot(x, theta).flatten()

def loss_elem_(y, y_hat):
    """
    Description:
    Calculates all the elements (y_pred - y)^2 of the loss function.
    Args:
    y: has to be an numpy.array, a vector.
    y_hat: has to be an numpy.array, a vector.
    Returns:
    J_elem: numpy.array, a vector of dimension (number of the training examples,1).
    None if there is a dimension matching problem between X, Y or theta.
    None if any argument is not of the expected type.
    Raises:
    This function should not raise any Exception.
    """
    if type(y_hat) != np.ndarray or type(y) != np.ndarray:
        return None
    if y.ndim == 1:
        y = y.reshape(y.size, 1)
    if y_hat.ndim == 1:
        y_hat = y_hat.reshape(y_hat.size, 1)
    if y.shape != y_hat.shape:
        return None
    result = []
    for i in range(len(y)):
        result.append(float((y_hat[i] - y[i])**2))
    return result

def loss_(y, y_hat):
    """
    Description:
    Calculates the value of loss function.
    Args:
    y: has to be an numpy.array, a vector.
    y_hat: has to be an numpy.array, a vector.
    Returns:
    J_value : has to be a float.
    None if there is a dimension matching problem between X, Y or theta.
    None if any argument is not of the expected type.
    Raises:
    This function should not raise any Exception.
    """
    if type(y_hat) != np.ndarray or type(y) != np.ndarray:
        return None
    if y.ndim == 1:
        y = y.reshape(y.size, 1)
    if y_hat.ndim == 1:
        y_hat = y_hat.reshape(y_hat.size, 1)
    if y.shape != y_hat.shape:
        return None
    result = (sum(loss_elem_(y, y_hat))) / (2 * y.size)
    return float(result)

