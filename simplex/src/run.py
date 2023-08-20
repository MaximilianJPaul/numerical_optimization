from active_set_method import active_set_method
from simplex_method_feasible import simplex
from simplex_method_infeasible import two_phase_simplex
import numpy as np
import time
from scipy.optimize import linprog
from problems_simplex import problems

def run_simplex_method():
    # Formate the result of the Simplex Method

    simplex_results = """Group: 36
Members: Maximilian Voshchepynets (k11946886)\n

Simplex Method Results on Feasible Starting Points\n\n
    """

    # Run the problems
    for i, problem in enumerate(problems):
        start_time = time.time()
        solution, iterations, status = simplex(problem['c'], problem['A'], problem['b'])
        elapsed_time = time.time() - start_time
        scipy_solution = linprog(c=problem['c'], A_ub=problem['A'], b_ub=problem['b']).x

        results = f"""
        ########## Problem {i + 1} ##########
        Method: Simplex
        Starting Point: Feasible

        Number of iterations: {iterations}
        Final Iterate: {solution}
        Stopping Criteria: {status}
        Real Solution: {scipy_solution}
        Running Time: {elapsed_time} seconds


        """

        simplex_results += results


    simplex_results += """\n\n\n
    Simplex Method Results on Infeasible Starting Points\n\n
    """


    for i, problem in enumerate(problems):
        i += 10
        A = problem["A"]
        b = problem["b"]
        c = problem["c"]
        x0 = problem["infeasible_point"]

        result = two_phase_simplex(A, b, c, x0)

        results = f"""
        ########## Problem {i + 1} ##########
        Method: Simplex
        Starting Point: Infeasible

        Number of iterations: {result['number_of_iterations']}
        Final Iterate: {result['final_iterate']}
        Stopping Criteria: {result['stopping_criteria']}
        Running Time: {result['running_time']} seconds


        """

        simplex_results += results
    
    with open('../report-simplex.txt', 'w') as file:
        file.write(simplex_results)


def run_active_set_method():
    active_set_results = f"""Group: 36
Members: Maximilian Voshchepynets (k11946886)\n

Active Set Method\n\n
    """

    # Running the Active Set Method on the QP reformulations
    m_values = np.arange(1, 6)
    n_values = 2 * m_values
    i = 1

    for m, n in zip(m_values, n_values):
        x0_values = [np.zeros(n) for _ in range(3)]
        for x0 in x0_values:
            M = np.random.randn(m, n)
            y = np.random.randn(m)
            y = y / np.linalg.norm(y) * np.linalg.norm(M, 2)  # ensure ||y|| >= ||M||_2

            # reformulate as QP
            G = M.T @ M
            c = -M.T @ y
            A = np.vstack([np.eye(n), -np.ones((1, n))])
            b = np.ones(n+1)

            # solve QP using active set method
            x, num_iter, running_time = active_set_method(G, c, A, b, x0)
            results = f"""
            ########## Problem {i} ##########
            Starting Point: {x0}

            m={m}, n={n}
            Solution: {x}
            Number of Iterations: {num_iter}
            Stopping criterion: Optimal solution found
            Running Time: {running_time} seconds
            

            """

            active_set_results += results

            i += 1




    # Define M and y
    M_tilde = np.array([[1, 1, 0, 0], [0, 0, 1, 1]])
    M = np.block([[np.diag(M_tilde[0, :2]), np.diag(M_tilde[0, 2:]), np.zeros((2, 16))], 
                [np.zeros((2, 4)), np.diag(M_tilde[1, :2]), np.diag(M_tilde[1, 2:]), np.zeros((2, 12))],
                [np.zeros((6, 20))]])
    y = np.array([1, -2, 3, -4, 5, -5, 4, -3, 2, -1])

    # Reformulate as QP
    G = M.T @ M + np.eye(20) * 1e-8  # Add regularization to G
    c = -M.T @ y
    A = np.vstack([np.eye(20), -np.ones((1, 20))])
    b = np.ones(21)

    # Define 5 starting points
    x0_values = [np.random.rand(20) for _ in range(5)]

    i = 16
    # Solve QP using active set method for each starting point
    for x0 in x0_values:
        x, num_iter, running_time = active_set_method(G, c, A, b, x0)

        # Compute the solution to the unconstrained problem
        x_unconstrained, residuals, rank, s = np.linalg.lstsq(M, y, rcond=None)

        result = f"""
            ########## Problem {i} ##########
            Starting Point: {x0}

            Solution: {x}
            Number of Iterations: {num_iter}
            Stopping criterion: Optimal solution found
            Running Time: {running_time} seconds
            Unconstrained Problem: {x_unconstrained}

        """

        active_set_results += result
        i += 1


    with open('../report-active-set-method.txt', 'w') as file:
        file.write(active_set_results)