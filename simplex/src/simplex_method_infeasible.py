import numpy as np
import time

def two_phase_simplex(A, b, c, x0):
    start = time.time()

    m, n = A.shape

    # Initialize x with correct length
    x = np.zeros(n)
    x[:len(x0)] = x0

    # Phase 1
    A_aux = np.hstack((A, np.eye(m)))
    c_aux = np.hstack((np.zeros(n), np.ones(m)))

    x, n_iterations_1 = phase_two_simplex(A_aux, b, c_aux, x)

    if np.any(x[n:] > 0):
        # No feasible solution found
        end = time.time()
        return {
            'number_of_iterations': n_iterations_1,
            'final_iterate': None,
            'stopping_criteria': 'No feasible solution',
            'running_time': end - start
        }

    # Phase 2
    A = A_aux[:,:n]
    x = x[:n]

    x, n_iterations_2 = phase_two_simplex(A, b, c, x)

    end = time.time()

    return {
        'number_of_iterations': n_iterations_1 + n_iterations_2,
        'final_iterate': x,
        'stopping_criteria': 'Optimal solution found',
        'running_time': end - start
    }


def phase_two_simplex(A, b, c, x, max_iterations=10000, epsilon=1e-9):
    m, n = A.shape

    n_iterations = 0
    while n_iterations < max_iterations:
        n_iterations += 1

        r = c - A.T @ (np.linalg.inv(A @ A.T) @ b)
        if np.all(r >= 0):
            break

        # Entering variable
        enter_var = np.argmin(r)

        temp = np.linalg.inv(A @ A.T) @ A @ np.eye(n,1,enter_var)
        if np.all(temp <= epsilon):
            break

        # Leaving variable
        ratios = np.full(m, np.inf)
        pos_indices = np.where(temp > epsilon)[0]
        ratios[pos_indices] = b[pos_indices] / temp[pos_indices]
        leave_var = np.argmin(ratios)

        # Pivot operation
        pivot = A[leave_var][enter_var]
        A[leave_var] /= pivot
        b[leave_var] /= pivot

        for i in range(m):
            if i == leave_var:
                continue
            factor = A[i][enter_var]
            A[i] -= factor * A[leave_var]
            b[i] -= factor * b[leave_var]

        x[enter_var], x[leave_var] = x[leave_var], x[enter_var]

    return x, n_iterations


    





