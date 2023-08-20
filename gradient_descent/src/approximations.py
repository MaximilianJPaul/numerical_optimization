import numpy as np

def gradient(x, f, h=1e-6):
    n = len(x)
    g = np.zeros(n)
    for i in range(n):
        x_plus = np.array(x, copy=True)
        x_plus[i] += h
        x_minus = np.array(x, copy=True)
        x_minus[i] -= h
        g[i] = (f(x_plus) - f(x_minus)) / (2 * h)
    return g


def hessian(x, f, h=1e-6):
    n = len(x)
    H = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            x_ij = np.array(x, copy=True)
            x_ij[i] += h
            x_ij[j] += h
            f_ij = f(x_ij)

            x_i = np.array(x, copy=True)
            x_i[i] += h
            f_i = f(x_i)

            x_j = np.array(x, copy=True)
            x_j[j] += h
            f_j = f(x_j)

            f_ = f(x)
            H[i, j] = (f_ij - f_i - f_j + f_) / (h ** 2)
    return H
