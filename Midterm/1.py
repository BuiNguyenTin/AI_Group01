import matplotlib.pyplot as plt
import numpy as np
import os

def visualize_matrix_with_matplotlib(file_paths):
    color_mapping = {
        'P': 'gold',  # yellow
        'E': 'green',  # green
        'G': 'pink',   # pink
        -1: 'black',    # black
        # white color is default, no need to specify
    }

    fig, axs = plt.subplots(1, len(file_paths), figsize=(12, 6))  # Create subplots

    for idx, file_path in enumerate(file_paths):
        matrix = []
        with open(file_path, 'r') as f:
            for line in f:
                row = eval(line.strip())  # Evaluate each line as a list
                matrix.append(row)

        nrows = len(matrix)
        ncols = len(matrix[0])

        # Create a matrix of zeros (we'll use this for plotting)
        plot_matrix = np.zeros((nrows, ncols))

        # Map values to colors
        for i in range(nrows):
            for j in range(ncols):
                value = matrix[i][j]
                if value in color_mapping:
                    plot_matrix[i][j] = -1 if value == -1 else 1  # use -1 for special characters and numbers
                elif isinstance(value, int):
                    plot_matrix[i][j] = 0  # default color (white) for other integers

        # Define custom colormap
        cmap = plt.cm.colors.ListedColormap(['black', 'white', 'green', 'pink'])
        bounds = [-1, 0, 1, 2, 3]
        norm = plt.cm.colors.BoundaryNorm(bounds, cmap.N)

        # Plot each matrix in a subplot
        axs[idx].imshow(plot_matrix, cmap=cmap, norm=norm, interpolation='nearest')

        # Customize ticks and grid
        axs[idx].set_xticks(np.arange(ncols))
        axs[idx].set_yticks(np.arange(nrows))
        axs[idx].set_xticklabels([])
        axs[idx].set_yticklabels([])

        # Annotate each cell with its value
        for i in range(nrows):
            for j in range(ncols):
                value = matrix[i][j]
                if value in color_mapping:
                    axs[idx].text(j, i, str(value), ha='center', va='center', fontsize=8, color='black')
                elif isinstance(value, int):
                    axs[idx].text(j, i, str(value), ha='center', va='center', fontsize=8, color='black')

        # Set title to file name without extension
        file_name = os.path.splitext(os.path.basename(file_path))[0]
        axs[idx].set_title(file_name, fontsize=12)

    # Adjust layout and display
    plt.tight_layout()
    plt.show()

# Usage example:
input_files = ['Floor_1.txt', 'Floor_2.txt']
visualize_matrix_with_matplotlib(input_files)
