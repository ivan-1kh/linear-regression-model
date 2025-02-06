import numpy as np

def compute_linear_regression(X, y, alpha=0.01, max_iterations=1000):
    #Flatten y array to 1 dimensional array
    y = y.flatten()
    n = X.shape[1]
    weights = np.zeros(n)

    #Set tolerance
    tolerance = 1e-6
    
    for i in range(max_iterations):
        gradients = gradient(X, y, weights)
        weights -= alpha * gradients
        new_cost = compute_cost(X, y, weights)
        
        # Convergence check
        gradient_magnitude = np.linalg.norm(gradients)
        if gradient_magnitude <= tolerance:
            break
    
    return weights, new_cost


def compute_cost(X, y, weights): 
    #Calculate the cost
    cost = 0
    predictions = np.dot(X, weights)
    residuals = y - predictions
    cost = np.sum(residuals**2) / (2 * len(y))
    return cost


def gradient(X, y, weights):
    #Calculate the bias and weight gradients
    predictions = np.dot(X, weights)
    residuals = y - predictions
    gradients = -2 * np.dot(X.T, residuals) / len(y)
    return gradients


def compute_r_squared(X, y, weights):
    #Compute predictions
    predictions = np.dot(X, weights)

    #Compute R squared
    residuals_fit = np.sum((y - predictions) ** 2)
    residuals_mean = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1 - (residuals_fit / residuals_mean)
    print("R squared: ", r_squared)