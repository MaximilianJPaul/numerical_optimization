import numpy as np

def backtracking_line_search(f, grad_f, x, direction, alpha=1, rho=0.5, c=1e-4):
    """
    Perform backtracking line search.
    
    Parameters:
        f: function to minimize
        grad_f: gradient of the function
        x: current point
        direction: search direction
        alpha: initial step size (default is 1)
        rho: factor to decrease alpha (default is 0.5)
        c: sufficient decrease constant (default is 1e-4)
        
    Returns:
        alpha: step size to use
    """
    while f(x + alpha * direction) > f(x) + c * alpha * np.dot(grad_f(x, f), direction):
        alpha *= rho
    return alpha


def exact_line_search(A, b, x, direction):
    # For linear systems, the exact line search solving A*x=b is alpha = (r'r)/(d'Ad)
    r = b - np.dot(A, x)
    alpha = np.dot(r, r) / np.dot(np.dot(direction, A), direction)
    return alpha