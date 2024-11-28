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
2. Polytope is bounded
3. Rak of A is n

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


input_file = "input1.csv"
MAX_ITER = 1000


def load_data(file_path):
    data_frame = pd.read_csv(file_path, header=None)

    matrix_A = data_frame.values[2:, :-1]
    vector_b = data_frame.values[2:, -1]
    cost_vector = data_frame.values[1, :-1]
    initial_z = data_frame.values[0, :-1]

    _, cols = matrix_A.shape

    if np.linalg.matrix_rank(matrix_A) != cols:
        raise np.linalg.LinAlgError("Matrix A is not of full rank:", matrix_A)

    return matrix_A, vector_b, cost_vector, initial_z


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


def find_next_vertex(matrix_A, vector_b, cost_vector, current_vertex):
    direction_matrix = compute_vertex_directions(matrix_A, vector_b, current_vertex)

    if direction_matrix is None:
        return False

    direction_costs = np.dot(direction_matrix, cost_vector)

    positive_cost_directions = []
    for i in range(len(direction_costs)):
        if direction_costs[i] > 0:
            positive_cost_directions.append(i)

    if len(positive_cost_directions) == 0:
        return True
    else:

        new_vertex = direction_matrix[positive_cost_directions[0]]

        unbounded = False
        for i in range(len(matrix_A)):
            if np.dot(matrix_A[i], new_vertex) > 0:
                unbounded = True
                break

        if not unbounded:
            raise np.linalg.LinAlgError("The Linear Program is unbounded.")

        untight_constraints = []
        for i in range(len(vector_b)):
            if not np.isclose(np.dot(matrix_A[i], current_vertex), vector_b[i]):
                untight_constraints.append(i)

        matrix_A2 = matrix_A[untight_constraints, :]
        vector_b2 = vector_b[untight_constraints]

        coefficients = []
        for i in range(len(vector_b2)):
            num = vector_b2[i] - np.dot(matrix_A2[i], current_vertex)
            denom = np.dot(matrix_A2[i], new_vertex)
            coefficients.append(num / denom)

        step_size = float("inf")
        for coeff in coefficients:
            if coeff >= 0 and coeff < step_size:
                step_size = coeff

        return current_vertex + step_size * new_vertex


def run_simplex(
    matrix_A,
    vector_b,
    cost_vector,
    start_vertex,
):
    iteration_steps = []

    max_iterations = MAX_ITER

    while max_iterations:
        iteration_steps.append([start_vertex, cost_vector.T @ start_vertex])
        next_vertex = find_next_vertex(matrix_A, vector_b, cost_vector, start_vertex)

        if isinstance(next_vertex, bool):
            return next_vertex, iteration_steps
        else:
            start_vertex = next_vertex

        max_iterations -= 1
    return False


def perturb_vector(vector):
    perturbed_vector = np.empty_like(vector)
    perturbed_vector[:] = vector
    return perturbed_vector


def main():
    matrix_A, vector_b, cost_vector, start_vertex = load_data(input_file)

    perturbed_b = perturb_vector(vector_b)

    for iteration in range(MAX_ITER):
        result, steps = run_simplex(matrix_A, perturbed_b, cost_vector, start_vertex)
        if result:
            print(f"Simplex algorithm converged in iteration {iteration + 1}.")
            for step in range(len(steps)):
                print(
                    "Iteration",
                    step + 1,
                    ": vertex =",
                    steps[step][0],
                    "cost =",
                    steps[step][1],
                )
            break
        else:
            print(f"Could not solve Linear Program.")


if __name__ == "__main__":
    main()
