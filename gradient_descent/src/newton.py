import numpy as np
from src.line_search import backtracking_line_search

def newton_method(f, grad_f, hessian_f, x0, eps=1e-6, max_iter=1000):
    x = x0

    for i in range(max_iter):
        gradient = grad_f(x, f)

        # Stop if the norm of the gradient is near 0
        if np.linalg.norm(gradient) < eps:
            break

        # Calculate the Newton direction
        hessian = hessian_f(x, f)
        newton_dir = -np.linalg.solve(hessian, gradient)

        # Perform line search
        alpha = backtracking_line_search(f, grad_f, x, newton_dir)
        x += alpha * newton_dir

    return x, i



def newton_method_modified(f, grad_f, hessian_f, x0, eps=1e-6, max_iter=1000):
    x = x0

    for i in range(max_iter):
        gradient = grad_f(x, f)

        # Stop if the norm of the gradient is near 0
        if np.linalg.norm(gradient) < eps:
            break

        # Calculate the Newton direction with Hessian modification
        hessian = hessian_f(x, f)

        # Add a multiple of the identity matrix to the Hessian
        # The multiplier is chosen to be a small positive number
        hessian_mod = hessian + 1e-3 * np.eye(len(x0))

        newton_dir = -np.linalg.solve(hessian_mod, gradient)

        # Perform line search
        alpha = backtracking_line_search(f, grad_f, x, newton_dir)
        x += alpha * newton_dir

    return x, i
