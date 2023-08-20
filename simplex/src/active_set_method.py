import numpy as np
from scipy.optimize import linprog
from time import time

def active_set_method(G, c, A, b, x0, tol=1e-6, max_iter=1000):
    m, n = A.shape
    x = x0.copy()
    W = np.where(A @ x - b < tol)[0]  # active set
    num_iter = 0
    start_time = time()

    while num_iter < max_iter:
        num_iter += 1
        # solve equality constrained QP
        G_W_inv = np.linalg.inv(G + A[W].T @ A[W])
        p = -G_W_inv @ (G @ x + A[W].T @ (A[W] @ x - b[W]))
        if np.linalg.norm(p) < tol:  # if p=0
            # compute Lagrange multipliers
            lambda_ = -G_W_inv @ A[W].T @ (A[W] @ x - b[W])
            if np.all(lambda_ >= 0):  # if all Lagrange multipliers are nonnegative
                break
            else:  # remove constraint with smallest Lagrange multiplier
                idx = np.argmin(lambda_)
                if idx < len(W):  # check if idx is a valid index for W
                    W = np.delete(W, idx)
        else:  # if p!=0
            alpha = 1
            for i in range(m):
                if i not in W and A[i] @ p > 0:
                    alpha_i = (b[i] - A[i] @ x) / (A[i] @ p)
                    if alpha_i < alpha:
                        alpha = alpha_i
                        j = i
            x += alpha * p
            if alpha < 1:  # add constraint j to active set
                W = np.append(W, j)

    end_time = time()
    running_time = end_time - start_time
    return x, num_iter, running_time