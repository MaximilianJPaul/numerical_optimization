import numpy as np
from src.line_search import backtracking_line_search

def fletcher_reeves(f, grad_f, x0, hessian_f=None, eps=1e-6, max_iter=1000):
    x = x0
    grad = grad_f(x, f)
    direction = -grad
    for i in range(max_iter):
        # Perform line search to find step size
        alpha = backtracking_line_search(f, grad_f, x, direction)
        
        # Update x
        x_new = x + alpha * direction
        
        # Check for convergence
        if np.linalg.norm(x_new - x) < eps:
            return x_new, i
        
        # Compute new gradient and conjugate direction
        grad_new = grad_f(x_new, f)
        beta = np.dot(grad_new, grad_new) / np.dot(grad, grad)
        direction = -grad_new + beta * direction
        
        # Update x and gradient for next iteration
        x = x_new
        grad = grad_new


    return x, i


def polak_ribiere(f, grad_f, x0, hessian_f=None, eps=1e-6, max_iter=1000):
    x = x0
    grad = grad_f(x, f)
    direction = -grad
    for i in range(max_iter):
        # Perform line search to find step size
        alpha = backtracking_line_search(f, grad_f, x, direction)
        
        # Update x
        x_new = x + alpha * direction
        
        # Check for convergence
        if np.linalg.norm(x_new - x) < eps:
            return x_new, i
        
        # Compute new gradient and conjugate direction
        grad_new = grad_f(x_new, f)
        beta = np.dot(grad_new, grad_new - grad) / np.dot(grad, grad)
        direction = -grad_new + max(0, beta) * direction  # Ensure beta is non-negative
        
        # Update x and gradient for next iteration
        x = x_new
        grad = grad_new

    return x



