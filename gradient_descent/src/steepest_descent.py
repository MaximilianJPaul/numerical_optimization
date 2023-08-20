import numpy as np
from src.line_search import exact_line_search

def steepest_descent(A, b, x0, eps=1e-6, max_iter=1000):
    x = x0
    for i in range(max_iter):
        r = b - np.dot(A, x)
        direction = r
        alpha = exact_line_search(A, b, x, direction)
        x_new = x + alpha * direction
        
        # Check for convergence
        if np.linalg.norm(b - np.dot(A, x_new)) < eps:
            return x_new
        
        # Update x for next iteration
        x = x_new

    return x, i