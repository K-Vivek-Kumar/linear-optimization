import csv
import numpy as np


def simplex_algorithm(csv_file):
    with open(csv_file, "r") as f:
        reader = csv.reader(f)
        data = list(reader)

    z = np.array([float(x) for x in data[0][:-1]])
    c = np.array([float(x) for x in data[1][:-1]])
    b = np.array([float(x) for x in [row[-1] for row in data[2:-1]]])
    A = np.array([[float(x) for x in row[:-1]] for row in data[2:-1]])

    m, n = A.shape

    assert np.linalg.matrix_rank(A) == n, "Rank of A is not equal to n."

    B = list(range(m))
    N = list(range(m, n))

    x_B = b.copy()
    full_solution = np.zeros(n)
    full_solution[B] = x_B

    print(f"Initial BFS: {full_solution}")
    print(f"Initial Objective Value: {np.dot(c, full_solution)}")

    num_iter = 0
    visited_vertices = set()
    vertex_sequence = []

    while True:

        A_B = A[:, B]
        A_N = A[:, N]

        c_B = c[B]
        c_N = c[N]

        B_inv = np.linalg.inv(A_B)
        reduced_costs = c_N - c_B @ B_inv @ A_N

        if np.all(reduced_costs >= 0):
            print("\nOptimal solution found.")
            break

        entering_index = np.argmin(reduced_costs)
        entering_variable = N[entering_index]

        direction = B_inv @ A[:, entering_variable]
        if np.all(direction <= 0):
            print("\nThe problem is unbounded.")
            break

        ratios = np.divide(
            x_B, direction, out=np.full_like(x_B, np.inf), where=direction > 0
        )
        leaving_index = np.argmin(ratios)
        leaving_variable = B[leaving_index]

        B[leaving_index] = entering_variable
        N[entering_index] = leaving_variable

        x_B = B_inv @ b
        full_solution = np.zeros(n)
        full_solution[B] = x_B

        objective_value = np.dot(c, full_solution)

        print(
            f"Iteration {num_iter + 1}: Vertex {full_solution}, Objective Value {objective_value}"
        )
        vertex_sequence.append((num_iter, full_solution.copy(), objective_value))

        rounded_vertex = tuple(np.round(full_solution, decimals=8))
        if rounded_vertex in visited_vertices:
            print(
                "\nPolytope degeneracy detected. The simplex method may cycle indefinitely."
            )
            break
        visited_vertices.add(rounded_vertex)

        num_iter += 1

    print("\nSequence of Vertices Visited:")
    for step, vertex, obj_value in vertex_sequence:
        print(f"Step {step}: Vertex {vertex}, Objective Value {obj_value}")

    print(f"\nTotal Iterations: {num_iter}")


simplex_algorithm("data.csv")
