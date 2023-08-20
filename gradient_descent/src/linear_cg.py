import numpy as np
from src.line_search import exact_line_search

def linear_cg(A, b, x0, eps=1e-6, max_iter=1000):
    x = x0
    r = b - np.dot(A, x)
    direction = r

    for i in range(max_iter):
        alpha = exact_line_search(A, b, x, direction)
        x_new = x + alpha * direction
        r_new = b - np.dot(A, x_new)
        
        # Check for convergence
        if np.linalg.norm(r_new) < eps:
            return x_new, i
        
        # Compute beta and update direction
        beta = np.dot(r_new, r_new) / np.dot(r, r)
        direction = r_new + beta * direction
        
        # Update residuals for next iteration
        r = r_new
        x = x_new

    return x, i