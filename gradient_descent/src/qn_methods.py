import numpy as np
from src.line_search import backtracking_line_search

def BFGS(f, grad_f, x0, hessian_f=None, eps=1e-6, max_iter=1000):
    # initialization
    x = x0
    n = len(x0)
    I = np.eye(n)
    H = I

    for i in range(max_iter):
        gradient = grad_f(x, f)
        
        # stop if the norm of the gradient is near 0
        if np.linalg.norm(gradient) < eps:
            break
            
        # calculate direction
        p = -np.dot(H, gradient)
        
        # perform line search
        alpha = backtracking_line_search(f, grad_f, x, p)
        
        # update x
        x_new = x + alpha * p
        s = x_new - x
        y = grad_f(x_new, f) - gradient
        
        # update H
        rho = 1.0 / np.dot(y, s)
        H = (I - rho * np.outer(s, y)) @ H @ (I - rho * np.outer(y, s)) + rho * np.outer(s, s)
        
        x = x_new

    return x, i



def SR1(f, grad_f, x0, hessian_f=None, eps=1e-6, max_iter=1000):
    # initialization
    x = x0
    n = len(x0)
    I = np.eye(n)
    H = I

    for i in range(max_iter):
        gradient = grad_f(x, f)
        
        # stop if the norm of the gradient is near 0
        if np.linalg.norm(gradient) < eps:
            break
            
        # calculate direction
        p = -np.dot(H, gradient)
        
        # perform line search
        alpha = backtracking_line_search(f, grad_f, x, p)
        
        # update x
        x_new = x + alpha * p
        s = x_new - x
        y = grad_f(x_new, f) - gradient
        
        # update H
        if np.dot((y - H @ s), s) != 0:
            H = H + np.outer(y - H @ s, y - H @ s) / np.dot((y - H @ s), s)
        
        x = x_new

    return x, i



def trust_region_sr1(f, grad_f, x0, hessian_f=None, delta=0.1, eta=0.2, eps=1e-6, max_iter=1000):
    # initialization
    x = x0
    n = len(x0)
    B = np.eye(n)  # initial approximation to the Hessian
    delta_max = 1.0

    for i in range(max_iter):
        gradient = grad_f(x, f)
        
        # stop if the norm of the gradient is near 0
        if np.linalg.norm(gradient) < eps:
            break

        # solve the trust-region subproblem to find p
        p = np.linalg.solve(B, -gradient)

        # ensure that p is within the trust region
        if np.linalg.norm(p) > delta:
            p = (delta / np.linalg.norm(p)) * p
        
        # evaluate the ratio
        try:
            rho = (f(x + p) - f(x)) / (-0.5 * p.dot(B).dot(p) - gradient.dot(p))
        except RuntimeWarning:
            print("Hello")
            rho = 0.0
        
        # update x and B
        if rho > eta:
            x = x + p
            s = p
            y = grad_f(x, f) - gradient
            
            # SR1 update
            Bs = B.dot(s)
            if np.dot(s - Bs, y - Bs) != 0:
                B = B + np.outer(y - Bs, y - Bs) / np.dot(y - Bs, s - Bs)
        
        # update trust-region radius
        if rho < 0.25:
            delta = 0.25 * delta
        elif rho > 0.75 and np.linalg.norm(p) == delta:
            delta = min(2 * delta, delta_max)

    return x, i
