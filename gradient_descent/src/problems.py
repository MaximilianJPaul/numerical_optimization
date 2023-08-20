import numpy as np

# Rosenbrock function
def rosenbrock(x):
    return 100 * (x[1] - x[0]**2)**2 + (1 - x[0])**2

def rosenbrock_grad(x, f=None):
    return np.array([-2*(1 - x[0]) - 400*x[0]*(x[1] - x[0]**2), 200*(x[1] - x[0]**2)])

def rosenbrock_hessian(x, f=None):
    return np.array([[2 - 400*x[1] + 1200*x[0]**2, -400*x[0]], [-400*x[0], 200]])


# Another function
def f(x):
    return 150 * (x[0] * x[1])**2 + (0.5 * x[0] + 2 * x[1] - 2)**2

def f_grad(x, f=None):
    return np.array([
        300 * x[0] * (x[1] ** 2) + 0.5 * x[0] + 2 * x[1] - 2,
        300 * x[1] * (x[0] ** 2) + 4 * (0.5 * x[0] + 2 * x[1] - 2)
    ])

def f_hessian(x, f=None):
    return np.array([
        [300 * (x[1] ** 2) + 0.5, 600 * x[0] * x[1] + 2],
        [600 * x[0] * x[1] + 2, 300 * (x[0] ** 2) + 8]
    ])


