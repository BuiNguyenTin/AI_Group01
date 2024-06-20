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

        # Perform DFS from 'P' to 'E' or from 'E' to 'G'
        if idx == 0:
            start_char, end_char = 'P', 'E'
            path_color = color_mapping['path']
        elif idx == 1:
            start_char, end_char = 'E', 'G'
            path_color = color_mapping['path']

        # Find starting point 'P' or 'E'
        start_point = None
        for i in range(nrows):
            for j in range(ncols):
                if matrix[i][j] == start_char:
                    start_point = (i, j)
                    break
            if start_point:
                break

        # Depth First Search (DFS) function
        def dfs(matrix, visited, i, j, end_char):
            # Base case: if current cell is end_char
            if matrix[i][j] == end_char:
                return [(i, j)]

            # Mark current cell as visited
            visited[i][j] = True

            # Define possible movements: right, left, down, up
            movements = [(0, 1), (0, -1), (1, 0), (-1, 0)]

            # Try each movement
            for move in movements:
                ni, nj = i + move[0], j + move[1]
                if 0 <= ni < nrows and 0 <= nj < ncols and not visited[ni][nj] and matrix[ni][nj] != -1:
                    path = dfs(matrix, visited, ni, nj, end_char)
                    if path:
                        return [(i, j)] + path

            # If no path found, backtrack
            return []

        # Initialize visited matrix
        visited = [[False] * ncols for _ in range(nrows)]

        # Perform DFS
        if start_point:
            path = dfs(matrix, visited, start_point[0], start_point[1], end_char)
            if path:
                for (i, j) in path:
                    axs[idx].text(j, i, matrix[i][j], ha='center', va='center', fontsize=8, color='black')
                    plot_matrix[i][j] = 5  # mark path in red
                    axs[idx].imshow(plot_matrix, cmap=cmap, norm=norm, interpolation='nearest')

        # Set title to file name without extension
        file_name = os.path.splitext(os.path.basename(file_path))[0]
        axs[idx].set_title(file_name, fontsize=12)

    # Adjust layout and display
    plt.tight_layout()
    plt.show()

# Usage example:
input_files = ['Floor_1.txt', 'Floor_2.txt']
visualize_matrix_with_path(input_files)
