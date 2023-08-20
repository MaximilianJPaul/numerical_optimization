import numpy as np

problems = [
    {
        "c": np.array([-1, -2]),
        "A": np.array([[1, 1], [2, 1]]),
        "b": np.array([2, 3]),
        "feasible_point": np.array([0, 0]),
        "infeasible_point": np.array([3, 3]),
    },
    {
        "c": np.array([-3, -2]),
        "A": np.array([[2, 1], [1, 3]]),
        "b": np.array([4, 3]),
        "feasible_point": np.array([0, 0]),
        "infeasible_point": np.array([3, 2]),
    },
    {
        "c": np.array([-1, -4]),
        "A": np.array([[2, 2], [1, 2]]),
        "b": np.array([4, 2]),
        "feasible_point": np.array([0, 0]),
        "infeasible_point": np.array([3, 3]),
    },
    {
        "c": np.array([-1, -1]),
        "A": np.array([[1, 3], [2, 1]]),
        "b": np.array([3, 2]),
        "feasible_point": np.array([0, 0]),
        "infeasible_point": np.array([2, 2]),
    },
    {
        "c": np.array([-2, -1]),
        "A": np.array([[3, 1], [1, 2]]),
        "b": np.array([3, 2]),
        "feasible_point": np.array([0, 0]),
        "infeasible_point": np.array([2, 2]),
    },
     {
        "c": np.array([-1, -2, -3, -4, -5, -6, -7, -8, -9, -10]),
        "A": np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]),
        "b": np.array([5, 30]),
        "feasible_point": np.zeros(10),
        "infeasible_point": np.ones(10),
    },
    {
        "c": np.array([-2, -1, -3, -1, -2, -3, -1, -2, -3, -1]),
        "A": np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]]),
        "b": np.array([55, 30]),
        "feasible_point": np.zeros(10),
        "infeasible_point": 2 * np.ones(10),
    },
    {
        "c": np.array([-3, -2, -1, -3, -2, -1, -3, -2, -1, -3]),
        "A": np.array([[1, 3, 5, 7, 9, 2, 4, 6, 8, 10], [10, 8, 6, 4, 2, 9, 7, 5, 3, 1]]),
        "b": np.array([60, 40]),
        "feasible_point": np.zeros(10),
        "infeasible_point": 3 * np.ones(10),
    },
    {
        "c": np.array([-1, -2, -1, -2, -1, -2, -1, -2, -1, -2]),
        "A": np.array([[10, 9, 8, 7, 6, 5, 4, 3, 2, 1], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]),
        "b": np.array([35, 60]),
        "feasible_point": np.zeros(10),
        "infeasible_point": 4 * np.ones(10),
    },
    {
        "c": np.array([-2, -2, -2, -2, -2, -2, -2, -2, -2, -2]),
        "A": np.array([[10, 8, 6, 4, 2, 10, 8, 6, 4, 2], [1, 3, 5, 7, 9, 1, 3, 5, 7, 9]]),
        "b": np.array([30, 70]),
        "feasible_point": np.zeros(10),
        "infeasible_point": 5 * np.ones(10),
    },
]