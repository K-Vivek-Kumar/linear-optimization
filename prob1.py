import numpy as np
import pandas as pd


def simplex_algorithm(csv_file):
    data = pd.read_csv(csv_file, header=None)
    z = data.iloc[0, :-1].values
    c = data.iloc[1, :-1].values
    A = data.iloc[2:-1, :-1].values
    b = data.iloc[2:-1, -1].values

    m, n = A.shape
    tableau = np.zeros((m + 1, n + 1))
    tableau[:-1, :-1] = A
    tableau[:-1, -1] = b
    tableau[-1, :-1] = -c
    tableau[-1, -1] = 0

    vertices_sequence = []
    objective_values = []

    iteration = 1
    while np.any(tableau[-1, :-1] < 0):
        pivot_column = np.argmin(tableau[-1, :-1])
        ratios = tableau[:-1, -1] / tableau[:-1, pivot_column]
        pivot_row = np.argmin(ratios)

        pivot_element = tableau[pivot_row, pivot_column]
        tableau[pivot_row, :] /= pivot_element
        for i in range(m + 1):
            if i != pivot_row:
                tableau[i, :] -= tableau[i, pivot_column] * tableau[pivot_row, :]

        current_vertex = tableau[:-1, -1].copy()
        objective_value = -tableau[-1, -1]
        print(
            f"Iteration {iteration}: Pivot Column={pivot_column}, Pivot Row={pivot_row}"
        )
        print(f"Current Vertex={current_vertex}, Objective Value={objective_value}\n")

        vertices_sequence.append(current_vertex)
        objective_values.append(objective_value)

        iteration += 1

    final_vertex = tableau[:-1, -1]
    final_objective_value = -tableau[-1, -1]

    print("Final Optimal Solution:")
    print("Vertices:", final_vertex)
    print("Objective Value:", final_objective_value)

    print("\nSequence of Vertices Visited:")
    for i, vertex in enumerate(vertices_sequence):
        print(
            f"Iteration {i + 1}: Vertex={vertex}, Objective Value={objective_values[i]}"
        )


simplex_algorithm("data.csv")
