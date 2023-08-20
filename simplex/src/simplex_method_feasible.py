import numpy as np
from scipy.optimize import linprog

def simplex(c, A, b):
    # Append slack variables to A, forming the initial tableau
    A = np.hstack([A, np.eye(A.shape[0])])
    c = np.hstack([c, np.zeros(A.shape[0])])

    # Append c and b to the tableau
    tableau = np.vstack([A, c])
    b = np.hstack([b, 0])
    tableau = np.column_stack([tableau, b])

    iterations = 0
    while True:
        # Choose pivot column: most negative entry in last row
        pivot_col = np.argmin(tableau[-1, :-1])

        if tableau[-1, pivot_col] >= 0:
            # All entries nonnegative - optimum found
            break

        # Choose pivot row: smallest nonnegative ratio in last column
        pivot_row = None
        min_ratio = np.inf
        for i in range(tableau.shape[0] - 1):
            if tableau[i, pivot_col] > 0:
                ratio = tableau[i, -1] / tableau[i, pivot_col]
                if ratio < min_ratio:
                    min_ratio = ratio
                    pivot_row = i

        # Perform pivot
        tableau[pivot_row, :] /= tableau[pivot_row, pivot_col]
        for i in range(tableau.shape[0]):
            if i != pivot_row:
                tableau[i, :] -= tableau[i, pivot_col] * tableau[pivot_row, :]

        iterations += 1

     # Create solution vector
    solution = np.zeros(len(c))
    for i in range(len(c)):
        col = tableau[:-1, i]
        one_row = np.where(col == 1)[0]
        if len(one_row) == 1:
            solution[i] = tableau[one_row, -1]

    return solution[:len(c) - A.shape[0]], iterations, "Optimal solution found"






