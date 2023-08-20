from src.problems import rosenbrock, rosenbrock_grad, rosenbrock_hessian, f, f_grad, f_hessian
from src.newton import newton_method, newton_method_modified
from src.cg_methods import fletcher_reeves, polak_ribiere
from src.linear_cg import linear_cg
from src.steepest_descent import steepest_descent
from src.qn_methods import BFGS, SR1, trust_region_sr1
from src.approximations import gradient, hessian
import numpy as np
from scipy.linalg import hilbert
import time

import warnings

# Ignore runtime warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)


intro = """
Group 36

Members: {
    "Maximilian Voshchepynets": k11946886
}

Solved Task: {
    1: True,
    2: True,
    3: True,
    4: True,
    5: True,
    6: True
}

"""

f_rosenbrock = {
    "name": "Function 1: Rosenbrock",
    "function": rosenbrock,
    "gradient": rosenbrock_grad,
    "hessian": rosenbrock_hessian,
    "initial_points": np.array([[1.2, 1.2], [-1.2, 1], [0.2, 0.8]]),
    "true_solution": np.array([1,1])
}

f_another = {
    "name": "Function 2",
    "function": f,
    "gradient": f_grad,
    "hessian": f_hessian,
    "initial_points": np.array([[-0.2, 1.2], [3.8, 0.1], [1.9, 0.6]]),
    "true_solution": np.array([[0, 1], [4, 0]])
}

algorithms = {
    "Newton Method": newton_method,
    "Newton Method with Hessian Modification": newton_method_modified,
    "The Fletcher-Reevers": fletcher_reeves,
    "The Polak Ribiere": polak_ribiere,
    "BFGS": BFGS,
    "SR1": SR1,
    "SR1 within the trust-region framework": trust_region_sr1
}

functions = [f_rosenbrock, f_another]


def single_run(i, method_name, method, f_name, f, grad, hess, x0, true_solution, grad_and_hess="True (not approximated)", eps=1e-6):
    x, num_iter = method(f=f, grad_f=grad, x0=x0.copy(), hessian_f=hess, eps=eps)

    result = f"""
    ########## Run {i} ##########
    Algorithm: {method_name}
    Function: {f_name}
    Initial Point: {x0}
    Gradient and Hessian: {grad_and_hess}

    Number of Iterations: {num_iter}
    Final Iterate x_k: {x}
    Size ||∇f(xk)||: {np.linalg.norm(grad(x, f))}
    Distance to the solution: {np.linalg.norm(x - true_solution) if len(true_solution) == 1 else min([np.linalg.norm(x - true_solution[0]), np.linalg.norm(x - true_solution[1])])}

    """

    return result


def run():
    i = 1
    result = ""

    # Run all the methods except Linear CG
    for method_name, method in algorithms.items():
        for function in functions:
            name = function["name"]
            f, grad, hess = function["function"], function["gradient"], function["hessian"]
            initial_points = function["initial_points"]
            true_solution = function["true_solution"]

            for x0 in initial_points:
                print(f"Task 1-5: Processed {i} runs", end="\r")
                true_result = single_run(i, method_name, method, name, f, grad, hess, x0, true_solution)
                i += 1

                print(f"Task 1-5: Processed {i} runs", end="\r")
                approximated_result = single_run(i, method_name, method, name, f, gradient, hessian, x0, true_solution, grad_and_hess="Approximated")
                i += 1

                result += true_result + approximated_result

    # Run Linear CG for n = [5, 8, 12, 20, 30]
    n_values = [5, 8, 12, 20, 30]
    for n in n_values:
        print(f"Task 1-5: Processed {i} runs", end="\r")
        A = hilbert(n)
        b = np.ones(n)
        x0 = np.zeros(n)

        x_cg, num_iter = linear_cg(A, b, x0)
        first = f"""
    ########## Run {i} ##########
    Algorithm: Linear CG
    n: {n}

    Number of Iterations: {num_iter}
    Final Iterate x_k: {x_cg}

        """
        i += 1
        
        print(f"Task 1-5: Processed {i} runs", end="\r")
        x_sd, num_iter = steepest_descent(A, b, x0)
        second = f"""
    ########## Run {i} ##########
    Algorithm: Steepest Descent
    n: {n}

    Number of Iterations: {num_iter}
    Final Iterate x_k: {x_sd}

        """
        i += 1

        result += first + second
    
    return result

def run_task_6():
    newton = algorithms["Newton Method"]
    BFGS = algorithms["BFGS"]

    result = ""

    i = 1
    print("")
    for function in functions:
            name = function["name"]
            f, grad, hess = function["function"], function["gradient"], function["hessian"]
            initial_points = function["initial_points"]

            for x0 in initial_points:
                # Newton's Method
                
                print(f"Task 6: Processed {i} runs", end='\r')

                start_time = time.time()
                x_newton, _ = newton(f, grad, hess, x0.copy())
                end_time = time.time()
                newton_time = end_time - start_time
                i += 1

                # BFGS
                print(f"Task 6: Processed {i} runs", end='\r')

                start_time = time.time()
                x_BFGS = BFGS(f, grad, x0.copy(), hess)
                end_time = time.time()
                BFGS_time = end_time - start_time

                res = f"""
    ########## Task 6 ##########
    
    Function to Optimize: {name}
    Initial Point: {x0}

    Time (Newton Method): {newton_time}
    Solution (Newton): {x_newton}

    Time (BFGS - Quasi-Newton): {BFGS_time}
    Solution (BFGS): {x_BFGS}

    Winner: {"Newton Method" if BFGS_time > newton_time else "BFGS"}

                """

                i += 1
                result += res
    
    result += """

    As we can see, for the second problem (not rosenbrock function), quesi-newton method BFGS converges faster than Newton Method.
    """

    return result

if __name__ == "__main__":
    with open('report.txt', 'w') as file:
        result = run()
        task_6 = run_task_6()

        file.write(intro)
        file.write(result)
        file.write(task_6)

        print('\nDone! Results are available in "./report.txt"')

