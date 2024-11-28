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
1. Rak of A is n

Implement the simplex algorithm to maximize the objective function, You need to implement the method discussed in class.

Input: CSV file with m+1 rows and n+1 column.
        The first row excluding the last element is the cost vector c of length n
        The last column excluding the top element is the constraint vector b of length m
        Rows two to m+1 and column one to n is the matrix A of size m*n

Output: You need to print the sequence of vertices visited and the value of the objective function at that vertex
"""


import numpy as np
import pandas as pd


input_file = "input4.csv"
MAX_ITER = 1000
EPSILON = 1e-4


def load_data(file_path):
    data_frame = pd.read_csv(file_path, header=None)

    matrix_A = data_frame.values[1:, :-1]
    vector_b = data_frame.values[1:, -1]
    cost_vector = data_frame.values[0, :-1]

    _, cols = matrix_A.shape

    if np.linalg.matrix_rank(matrix_A) != cols:
        raise np.linalg.LinAlgError("A is not of full rank:", matrix_A)

    return matrix_A, vector_b, cost_vector


def get_initial_vertex(
    constraint_matrix,
    rhs_vector,
) -> np.ndarray:
    row_count, col_count = constraint_matrix.shape
    random_generator = np.random.default_rng()

    max_times = MAX_ITER

    for _ in range(max_times):
        selected_rows = random_generator.choice(row_count, col_count, replace=False)
        selected_matrix = constraint_matrix[selected_rows]
        selected_rhs = rhs_vector[selected_rows]

        if np.linalg.matrix_rank(selected_matrix) == col_count:
            candidate_vertex = np.linalg.inv(selected_matrix) @ selected_rhs

            if np.all(constraint_matrix @ candidate_vertex <= rhs_vector):
                return candidate_vertex

    raise RuntimeError("Unable to find")


def apply_perturbation(original_b):
    random_generator = np.random.default_rng()
    perturbation_values = random_generator.uniform(
        EPSILON, 2 * EPSILON, original_b.shape
    )
    perturbed_b = original_b + perturbation_values
    return perturbed_b


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


def find_direction_to_next_vertex(
    constraint_matrix,
    constraint_vector,
    objective_coefficients,
    current_vertex,
):
    directions = compute_vertex_directions(
        constraint_matrix, constraint_vector, current_vertex
    )

    if directions is None:
        return False

    direction_costs = directions @ objective_coefficients
    positive_cost_directions = np.where(direction_costs > 0)[0]

    if len(positive_cost_directions) == 0:
        return True
    else:
        next_direction = directions[positive_cost_directions[0]]

        if len(np.where(constraint_matrix @ next_direction > 0)[0]) == 0:
            raise np.linalg.LinAlgError("Linear program is unbounded.")

        non_tight_constraints = np.where(
            ~np.isclose(constraint_matrix @ current_vertex, constraint_vector)
        )
        reduced_matrix = constraint_matrix[non_tight_constraints]
        reduced_vector = constraint_vector[non_tight_constraints]

        step_coefficients = (reduced_vector - reduced_matrix @ current_vertex) / (
            reduced_matrix @ next_direction
        )
        step_size = np.min(step_coefficients[step_coefficients >= 0])

        return current_vertex + step_size * next_direction


def run_simplex_algorithm(
    constraint_matrix,
    constraint_vector,
    objective_coefficients,
    current_vertex,
):
    history_of_steps = []
    max_iterations = MAX_ITER

    while max_iterations:
        history_of_steps.append(
            [current_vertex, objective_coefficients.T @ current_vertex]
        )
        next_vertex = find_direction_to_next_vertex(
            constraint_matrix, constraint_vector, objective_coefficients, current_vertex
        )

        if isinstance(next_vertex, bool):
            return next_vertex, history_of_steps
        else:
            current_vertex = next_vertex
        max_iterations -= 1

    return False


def main():
    constraint_matrix, constraint_vector, objective_coefficients = load_data(input_file)
    perturbed_constraint_vector = np.empty_like(constraint_vector)
    perturbed_constraint_vector[:] = constraint_vector
    for i in range(MAX_ITER):
        starting_vertex = get_initial_vertex(
            constraint_matrix, perturbed_constraint_vector
        )
        is_solution_found, history_of_steps = run_simplex_algorithm(
            constraint_matrix,
            perturbed_constraint_vector,
            objective_coefficients,
            starting_vertex,
        )
        if is_solution_found:
            print(f"Simplex algorithm converged in iteration {i+1}.")
            for j in range(len(history_of_steps)):
                print(
                    f"Iteration {j+1}: x = {history_of_steps[j][0]}, cost = {history_of_steps[j][1]}"
                )
            break
        else:
            perturbed_constraint_vector = apply_perturbation(constraint_vector)
            print(f"Degeneracy detected in iteration {i+1}. LP has been perturbed.")


if __name__ == "__main__":
    main()
