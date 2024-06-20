import heapq
import matplotlib.pyplot as plt
import numpy as np
import os
import psutil
import time

def load_matrix_from_file(file_path):
    matrix = []
    with open(file_path, 'r') as f:
        for line in f:
            row = eval(line.strip())  # Evaluate each line as a list
            matrix.append(row)
    return matrix

def heuristic_cost_estimate(start, end):
    return abs(start[0] - end[0]) + abs(start[1] - end[1])

def astar(matrix, start_char, end_char):
    if not matrix:
        return "Matrix is empty"

    nrows = len(matrix)
    ncols = len(matrix[0])

    # Find starting and ending points
    start_point = None
    end_point = None
    for i in range(nrows):
        for j in range(ncols):
            if matrix[i][j] == start_char:
                start_point = (i, j)
            elif matrix[i][j] == end_char:
                end_point = (i, j)

    if not start_point or not end_point:
        return None  # No path found

    # Directions: right, left, down, up
    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]

    # Priority queue for A* algorithm
    pq = []
    heapq.heappush(pq, (0, start_point, []))  # (f_cost, current_node, path)

    # Initialize visited set and cost dictionary
    visited = set()
    g_cost = {start_point: 0}
    f_cost = {start_point: heuristic_cost_estimate(start_point, end_point)}

    # Perform A* search
    while pq:
        current_f_cost, (current_i, current_j), path = heapq.heappop(pq)

        # Check if we reached the end point
        if (current_i, current_j) == end_point:
            return path + [(current_i, current_j)]

        # Mark current node as visited
        visited.add((current_i, current_j))

        # Explore neighbors
        for di, dj in directions:
            ni, nj = current_i + di, current_j + dj
            if 0 <= ni < nrows and 0 <= nj < ncols and (ni, nj) not in visited and matrix[ni][nj] != -1:
                new_g_cost = g_cost[(current_i, current_j)] + 1
                if (ni, nj) not in g_cost or new_g_cost < g_cost[(ni, nj)]:
                    g_cost[(ni, nj)] = new_g_cost
                    f_cost[(ni, nj)] = new_g_cost + heuristic_cost_estimate((ni, nj), end_point)
                    heapq.heappush(pq, (f_cost[(ni, nj)], (ni, nj), path + [(current_i, current_j)]))

    return None  # If no path found

def visualize_matrix_with_path(file_paths):
    color_mapping = {
        'P': 'gold',    # yellow
        'E': 'lightgreen',  # green
        'G': 'pink',    # pink
        -1: 'black',    # black
        0: 'white',     # white
        'path': 'red'   # red for path
    }

    fig, axs = plt.subplots(1, len(file_paths), figsize=(12, 6))  # Create subplots

    total_execution_time = 0.0
    total_memory_usage = 0.0

    for idx, file_path in enumerate(file_paths):
        matrix = load_matrix_from_file(file_path)
        nrows = len(matrix)
        ncols = len(matrix[0])

        # Create a matrix of zeros (we'll use this for plotting)
        plot_matrix = np.zeros((nrows, ncols))

        # Map values to colors
        for i in range(nrows):
            for j in range(ncols):
                value = matrix[i][j]
                if isinstance(value, int):
                    if value == -1:
                        plot_matrix[i][j] = 3  # black for -1
                    else:
                        plot_matrix[i][j] = 0  # white for other integers
                else:
                    plot_matrix[i][j] = 1 if value == 'P' else 2  # yellow for 'P', green for 'E', pink for 'G'

        # Define custom colormap
        cmap = plt.cm.colors.ListedColormap(['white', 'gold', 'lightgreen', 'black', 'pink', 'red'])
        bounds = [0, 1, 2, 3, 4, 5]
        norm = plt.cm.colors.BoundaryNorm(bounds, cmap.N)

        # Plot the matrix
        axs[idx].imshow(plot_matrix, cmap=cmap, norm=norm, interpolation='nearest')

        # Customize ticks and grid
        axs[idx].set_xticks(np.arange(ncols))
        axs[idx].set_yticks(np.arange(nrows))
        axs[idx].set_xticklabels([])
        axs[idx].set_yticklabels([])

        # Perform A* from 'P' to 'E' or from 'E' to 'G'
        if idx == 0:
            start_char, end_char = 'P', 'E'
            path_color = color_mapping['path']
        elif idx == 1:
            start_char, end_char = 'E', 'G'
            path_color = color_mapping['path']

        # Measure memory usage before A*
        start_memory_usage = psutil.Process().memory_info().rss / 1024.0  # in KB

        # Measure execution time
        start_time = time.time()

        # Find path using A*
        path = astar(matrix, start_char, end_char)
        if path:
            for (i, j) in path:
                axs[idx].text(j, i, matrix[i][j], ha='center', va='center', fontsize=8, color='black')
                plot_matrix[i][j] = 5  # mark path in red
            axs[idx].imshow(plot_matrix, cmap=cmap, norm=norm, interpolation='nearest')

        # Measure execution time and memory usage after A*
        end_time = time.time()
        end_memory_usage = psutil.Process().memory_info().rss / 1024.0  # in KB

        execution_time = end_time - start_time
        memory_usage = end_memory_usage - start_memory_usage

        total_execution_time += execution_time
        total_memory_usage += memory_usage

        # Set title to file name without extension
        file_name = os.path.splitext(os.path.basename(file_path))[0]
        axs[idx].set_title(file_name, fontsize=12)

    # Print total execution time and memory usage
    print(f"Total Execution Time: {total_execution_time:.6f} seconds")
    print(f"Total Memory Usage: {total_memory_usage:.2f} KB")

    # Adjust layout and display
    plt.tight_layout()
    plt.show()

# Usage example:
input_files = ['Floor_1.txt', 'Floor_2.txt']
visualize_matrix_with_path(input_files)
