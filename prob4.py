import numpy as np
import pandas as pd


def simplex_algorithm(csv_file, max_iterations=1000):

    data = pd.read_csv(csv_file, header=None)
    c = data.iloc[0, :-1].values
    A = data.iloc[1:-1, :-1].values
    b = data.iloc[1:-1, -1].values

    m, n = A.shape

    tableau = np.zeros((m + 1, n + 1))
    tableau[:-1, :-1] = A
    tableau[:-1, -1] = b
    tableau[-1, :-1] = -c
    tableau[-1, -1] = 0

    iteration = 1
    visited_vertices = set()
    vertex_sequence = []

    while iteration <= max_iterations:

        pivot_col = np.argmin(tableau[-1, :-1])

        pivot_col_values = tableau[:-1, pivot_col]

        if np.all(pivot_col_values <= 0):
            print("The problem is infeasible.")
            return

        ratios = tableau[:-1, -1] / np.where(
            pivot_col_values > 0, pivot_col_values, np.inf
        )

        pivot_row = np.argmin(ratios)

        pivot_element = tableau[pivot_row, pivot_col]
        tableau[pivot_row, :] /= pivot_element
        for i in range(m + 1):
            if i != pivot_row:
                tableau[i, :] -= tableau[i, pivot_col] * tableau[pivot_row, :]

        current_vertex = tuple(tableau[:-1, -1])
        objective_value = -tableau[-1, -1]
        print(
            f"Iteration {iteration}: Vertex={current_vertex}, Objective Value={objective_value}"
        )

        vertex_sequence.append((iteration, current_vertex, objective_value))

        if current_vertex in visited_vertices:
            print(
                "Polytope Degeneracy Detected. The simplex method may cycle indefinitely."
            )
            break
        visited_vertices.add(current_vertex)

        iteration += 1

    if iteration > max_iterations:
        print("The problem may be unbounded. Reached the maximum number of iterations.")

    final_vertex = tableau[:-1, -1]
    final_objective_value = -tableau[-1, -1]

    print("\nFinal Optimal Solution:")
    print("Vertices:", final_vertex)
    print("Objective Value:", final_objective_value)

    print("\nSequence of Vertices Visited:")
    for step, vertex, obj_value in vertex_sequence:
        print(f"Step {step}: Vertex {vertex}, Objective Value {obj_value}")


simplex_algorithm("data.csv", max_iterations=1000)
