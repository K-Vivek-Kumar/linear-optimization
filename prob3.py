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
1. Rank of A is n

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


input_file = "input3.csv"
MAX_ITER = 1000
EPSILON = 1e-4


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


def find_next_vertex_using_simplex_algorithm(
    constraint_matrix, right_hand_side_vector, cost_vector, current_vertex
):
    direction_matrix = compute_vertex_directions(
        constraint_matrix, right_hand_side_vector, current_vertex
    )

    if direction_matrix is None:
        return False

    direction_costs = direction_matrix @ cost_vector
    positive_direction_indices = np.where(direction_costs > 0)[0]

    if len(positive_direction_indices) == 0:
        return True
    else:
        chosen_direction = direction_matrix[positive_direction_indices[0]]

        unbounded_check = np.where(constraint_matrix @ chosen_direction > 0)[0]
        if len(unbounded_check) == 0:
            raise np.linalg.LinAlgError(
                "Linear Program is unbounded in the direction of the chosen vertex."
            )

        non_tight_constraints = np.where(
            ~np.isclose(constraint_matrix @ current_vertex, right_hand_side_vector)
        )
        constraint_matrix_for_non_tight = constraint_matrix[non_tight_constraints]
        right_hand_side_for_non_tight = right_hand_side_vector[non_tight_constraints]

        coefficients_for_step_size = (
            right_hand_side_for_non_tight
            - constraint_matrix_for_non_tight @ current_vertex
        ) / (constraint_matrix_for_non_tight @ chosen_direction)
        valid_coefficients = coefficients_for_step_size[coefficients_for_step_size >= 0]
        step_size = np.min(valid_coefficients)

        next_vertex = current_vertex + step_size * chosen_direction
        return next_vertex


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
        next_vertex = find_next_vertex_using_simplex_algorithm(A, b, c, initial_vertex)

        if isinstance(next_vertex, bool):
            return next_vertex, steps
        else:
            initial_vertex = next_vertex

        max_iterations -= 1

    return False


def main():
    constraint_matrix, original_rhs, cost_vector, initial_vertex = load_data(input_file)

    if not np.all(constraint_matrix @ initial_vertex <= original_rhs):
        print("Initial vertex is outside the feasible region.")
        exit(-1)
    rhs_with_perturbation = np.empty_like(original_rhs)
    rhs_with_perturbation[:] = original_rhs
    for iteration in range(MAX_ITER):
        is_optimal, path_steps = run_simplex_algorithm(
            constraint_matrix, rhs_with_perturbation, cost_vector, initial_vertex
        )
        if is_optimal:
            print(f"Simplex algorithm converged in iteration {iteration + 1}.")
            for step in range(len(path_steps)):
                print(
                    f"Step {step + 1}: vertex = {path_steps[step][0]}, cost = {path_steps[step][1]}"
                )
            break
        else:
            rhs_with_perturbation = apply_perturbation(original_rhs)
            print(
                f"Degeneracy detected in iteration {iteration + 1}. Perturbing the right-hand side of the LP."
            )


if __name__ == "__main__":
    main()
