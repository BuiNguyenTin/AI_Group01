import matplotlib.pyplot as plt
import numpy as np
import os

def load_matrix_from_file(file_path):
    matrix = []
    with open(file_path, 'r') as f:
        for line in f:
            row = eval(line.strip())  # Evaluate each line as a list
            matrix.append(row)
    return matrix

def hill_climbing(matrix, start_char, end_char):
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

    # Initialize current position and path
    current_i, current_j = start_point
    current_path = [(current_i, current_j)]
    current_value = 0

    # Directions: right, left, down, up
    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]

    # Perform hill climbing
    while (current_i, current_j) != end_point:
        next_move = None
        next_value = -float('inf')

        # Explore neighbors
        for di, dj in directions:
            ni, nj = current_i + di, current_j + dj
            if 0 <= ni < nrows and 0 <= nj < ncols and matrix[ni][nj] != -1:
                neighbor_value = matrix[ni][nj]
                # Check if neighbor_value is convertible to int
                if isinstance(neighbor_value, int):
                    value_to_compare = neighbor_value
                else:
                    try:
                        value_to_compare = int(neighbor_value)
                    except ValueError:
                        continue  # Skip this neighbor if conversion fails

                if value_to_compare > next_value:
                    next_value = value_to_compare
                    next_move = (ni, nj)

        # Move to the best neighbor
        if next_move:
            current_i, current_j = next_move
            current_path.append(next_move)
            current_value += next_value  # Accumulate the value

    return current_path, current_value

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

        # Perform hill climbing from 'P' to 'E' or from 'E' to 'G'
        if idx == 0:
            start_char, end_char = 'P', 'E'
            path_color = color_mapping['path']
        elif idx == 1:
            start_char, end_char = 'E', 'G'
            path_color = color_mapping['path']

        # Find path using hill climbing
        path, total_value = hill_climbing(matrix, start_char, end_char)
        if path:
            for (i, j) in path:
                axs[idx].text(j, i, matrix[i][j], ha='center', va='center', fontsize=8, color='black')
                plot_matrix[i][j] = 5  # mark path in red
                axs[idx].imshow(plot_matrix, cmap=cmap, norm=norm, interpolation='nearest')

            # Print total value for the path
            print(f"Total value for path {idx+1}: {total_value}")

        # Set title to file name without extension
        file_name = os.path.splitext(os.path.basename(file_path))[0]
        axs[idx].set_title(file_name, fontsize=12)

    # Adjust layout and display
    plt.tight_layout()
    plt.show()

# Usage example:
input_files = ['Floor_1.txt', 'Floor_2.txt']
visualize_matrix_with_path(input_files)
