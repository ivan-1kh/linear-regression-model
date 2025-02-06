import numpy as np

def compute_cost(X, y, weights):
    """
    Compute the cost function (mean squared error) for linear regression.
    
    Parameters:
        X (numpy.ndarray): The data matrix of shape (m, n+1) which includes a bias column.
        y (numpy.ndarray): The target values of shape (m, 1).
        weights (numpy.ndarray): The weights vector of shape (n+1, 1).
    
    Returns:
        float: The computed cost.
    """
    m = y.shape[0]  # number of training examples
    predictions = np.dot(X, weights)  # shape (m, 1)
    errors = predictions - y          # shape (m, 1)
    cost = (1 / (2 * m)) * np.sum(np.square(errors))
    return cost

def gradient(X, y, weights):
    """
    Compute the gradient of the cost function for linear regression.
    
    Parameters:
        X (numpy.ndarray): The data matrix of shape (m, n+1) which includes a bias column.
        y (numpy.ndarray): The target values of shape (m, 1).
        weights (numpy.ndarray): The weights vector of shape (n+1, 1).
    
    Returns:
        numpy.ndarray: The gradient, an array of shape (n+1, 1).
    """
    m = y.shape[0]  # number of training examples
    predictions = np.dot(X, weights)  # shape (m, 1)
    errors = predictions - y          # shape (m, 1)
    grad = (1 / m) * np.dot(X.T, errors)
    return grad

def compute_linear_regression(X, y, alpha=0.01, max_iterations=1000):
    """
    Compute linear regression using gradient descent.
    
    This function adds a bias term to the input data, initializes the weights,
    and then performs gradient descent to optimize the weights.
    
    Parameters:
        X (numpy.ndarray): The data matrix of shape (m, n) where m is the number of examples
                           and n is the number of features.
        y (numpy.ndarray): The target values of shape (m, 1).
        alpha (float, optional): The learning rate for gradient descent. Default is 0.01.
        max_iterations (int, optional): The maximum number of iterations for gradient descent.
                                        Default is 1000.
    
    Returns:
        tuple: A tuple containing:
            - weights (numpy.ndarray): A 1-D array of shape (n+1,) with the learned weights,
                                       where the first element is the bias term.
            - final_cost (float): The value of the cost function after the final iteration.
    """
    m = X.shape[0]
    
    # Add a column of ones to X to incorporate the bias term.
    X_bias = np.hstack((np.ones((m, 1)), X))
    
    # Initialize weights as a column vector (n+1, 1)
    n = X_bias.shape[1]
    weights = np.zeros((n, 1))
    
    # Run gradient descent
    for i in range(max_iterations):
        grad = gradient(X_bias, y, weights)
        weights = weights - alpha * grad
    
    final_cost = compute_cost(X_bias, y, weights)
    
    # Return weights as a 1-D array and the final cost.
    return (weights.flatten(), final_cost)

def compute_r_squared(X, y, weights):
    #Compute predictions
    predictions = np.dot(X, weights) 

    #Compute R squared
    residuals_fit = np.sum((y - predictions) ** 2)  
    residuals_mean = np.sum((y - np.mean(y)) ** 2)  
    r_squared = 1 - (residuals_fit / residuals_mean)
    print("R squared: ", r_squared)
