# This code is submitted as a part of Linear Optimisation Assignment Problem 1.
#
# Team Members:
#   Bhukya Roopak Krishna (MS21BTECH11012)
#   K Vivek Kumar (CS21BTECH11026)
#   Balaji Tanushree Singh (ES21BTECH11009)
#
# Github Repo: https://github.com/K-Vivek-Kumar/linear-optimization.git


"""
Assumption
1. Polytope is non-degenerate.
2. Rak of A is n

Implement the simplex algorithm to maximize the objective function, You need to implement the method discussed in class.

Input: CSV file with m+2 rows and n+1 column.
        The first row excluding the last element is the initial feasible point z of length n
        The second row excluding the last element is the cost vector c of length n
        The last column excluding the top two elements is the constraint vector b of length m
        Rows third to m+2 and column one to n is the matrix A of size m*n

Output: You need to print the sequence of vertices visited and the value of the objective function at that vertex
"""


import numpy as np
import pandas as pd


input_file = "input2.csv"
MAX_ITER = 1000


def load_data(file_path):
    data_frame = pd.read_csv(file_path, header=None)

    matrix_A = data_frame.values[2:, :-1]
    vector_b = data_frame.values[2:, -1]
    cost_vector = data_frame.values[1, :-1]
    initial_X = data_frame.values[0, :-1]

    _, cols = matrix_A.shape

    if np.linalg.matrix_rank(matrix_A) != cols:
        raise np.linalg.LinAlgError("A is not of full rank:", matrix_A)

    return matrix_A, vector_b, cost_vector, initial_X


def compute_vertex_directions(matrix_A, vector_b, vertex):
    tight_constraints = []
    for i in range(len(vector_b)):
        if np.isclose(matrix_A[i] @ vertex, vector_b[i]):
            tight_constraints.append(i)

    if not tight_constraints:
        return None

    matrix_A1 = matrix_A[tight_constraints, :]

    if matrix_A1.shape[0] == matrix_A1.shape[1]:
        return -np.linalg.inv(matrix_A1.T)
    else:
        return None


def find_neighbouring_vertex(
    A,
    b,
    c,
    current_vertex,
):
    direction_matrix = compute_vertex_directions(A, b, current_vertex)
    if direction_matrix is None:
        return False

    direction_costs = direction_matrix @ c
    positive_cost_directions = np.where(direction_costs > 0)[0]

    if len(positive_cost_directions) == 0:
        return True
    else:
        new_direction = direction_matrix[positive_cost_directions[0]]
        if len(np.where(A @ new_direction > 0)[0]) == 0:
            raise np.linalg.LinAlgError("Linear Program is unbounded.")

        untight_rows = np.where(~np.isclose(A @ current_vertex, b))
        A_untight = A[untight_rows]
        b_untight = b[untight_rows]

        coefficients = (b_untight - A_untight @ current_vertex) / (
            A_untight @ new_direction
        )
        step_size = np.min(coefficients[coefficients >= 0])

        return current_vertex + step_size * new_direction


def run_simplex_algorithm(
    A,
    b,
    c,
    initial_vertex,
):
    steps = []
    max_iterations = MAX_ITER

    while max_iterations:
        steps.append([initial_vertex, c.T @ initial_vertex])
        next_vertex = find_neighbouring_vertex(A, b, c, initial_vertex)

        if isinstance(next_vertex, bool):
            return next_vertex, steps
        else:
            initial_vertex = next_vertex

        max_iterations -= 1

    return False


def main():
    A_matrix, b_vector, cost_vector, initial_vertex = load_data(input_file)

    perturbed_b_vector = np.empty_like(b_vector)
    perturbed_b_vector[:] = b_vector
    for iteration in range(MAX_ITER):
        is_optimal, solution_steps = run_simplex_algorithm(
            A_matrix, perturbed_b_vector, cost_vector, initial_vertex
        )
        if is_optimal:
            print(f"Simplex algorithm converged in iteration {iteration + 1}.")

            for step_index in range(len(solution_steps)):
                print(
                    "Iteration",
                    step_index + 1,
                    ": vertex =",
                    solution_steps[step_index][0],
                    "cost =",
                    solution_steps[step_index][1],
                )
            break
        else:
            print(f"Could not solve the linear program.")


if __name__ == "__main__":
    main()
