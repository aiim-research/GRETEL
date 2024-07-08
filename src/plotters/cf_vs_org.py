import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from jsonpickle import decode
import os

from src.core.plotter_base import Plotter

class CfVSOrg(Plotter):
    
    def plot(self, read_path):
        filepaths = list()
        for subdir, _, files in os.walk(read_path):
            for file in files:
                filepath = os.path.join(subdir, file)
                filepaths.append(filepath)
        
        pairs = {i: {} for i in range(self.dataset.num_classes)}
    
        for file in filepaths:
            with open(file, 'r') as f:
                data = decode(f.read())
                original_id = data['original_id']
                
                cf = data['counterfactual_adj']
                instance = self.dataset.get_instance(original_id)
                original = instance.data
                
                if not data['correctness']:
                    cf = np.array([])
                    
                pairs[instance.label].update({original_id: [original, cf]})
            
        all_changes = {i: [] for i in range(self.dataset.num_classes)}
        for label in pairs:
            for id in pairs[label]:
                changes = self.__changes(pairs[label][id][0], pairs[label][id][1])
                all_changes[label].append(changes)
            
        self.plot_matrices_in_row(all_changes, 
                                  save_path=os.path.join(self.dump_path, self.dataset.name, 
                                                         self.oracle.name, self.explainer.name))
    
    
    def __changes(self, A, B):
        if not len(B):
            changes_matrix = np.ones((A.shape[0], A.shape[1], 3), dtype=np.uint8)
            changes_matrix *= 255
        else:
            n = max(A.shape[0], B.shape[0])
            changes_matrix = np.zeros((n, n, 3), dtype=np.uint8)
        
            added_cells = np.logical_and(B == 1, A == 0)
            unchanged_cells_A = np.logical_and(B == 0, A == 0)
            deleted_cells = np.logical_and(B == 0, A == 1)
                        
            for i in range(n):
                for j in range(n):
                    if added_cells[i, j]:
                        changes_matrix[i, j, :] = [129, 206, 3]  # Green for added cells
                    elif deleted_cells[i, j]:
                        changes_matrix[i, j, :] = [226, 4, 4]  # Red for deleted cells
                    elif unchanged_cells_A[i, j]:
                        changes_matrix[i, j, :] = [255, 255, 255] 
                    elif A[i, j] == 1:
                        changes_matrix[i, j, :] = [0, 0, 0]
                        
        return changes_matrix

    def plot_matrices_in_row(self, matrices, save_path=None):
        lengths = list(map(len, matrices.values()))
        max_instances = max(lengths)
        num_images_per_row = min(10, max_instances)  # Adjust the number of columns based on the maximum instances

        # Calculate the total number of rows needed for all images
        total_rows = 0
        for length in lengths:
            total_rows += (length + num_images_per_row - 1) // num_images_per_row
        total_rows += len(lengths) - 1  # Additional rows for the horizontal lines between groups

        fig_height = total_rows * 2  # Adjust figure height based on total rows
        fig = plt.figure(figsize=(2 * num_images_per_row, fig_height))
        gs = GridSpec(total_rows, num_images_per_row, figure=fig, wspace=.05, hspace=.03)

        current_row = 0
        for label, matrix_list in matrices.items():
            num_rows_for_label = (len(matrix_list) + num_images_per_row - 1) // num_images_per_row
            for i, change_matrix in enumerate(matrix_list):
                row = current_row + (i // num_images_per_row)
                col = i % num_images_per_row
                ax = fig.add_subplot(gs[row, col])
                ax.imshow(change_matrix, vmin=0, vmax=255)
                ax.set_xticks([])  # Hide x-axis ticks
                ax.set_yticks([])  # Hide y-axis ticks

            current_row += num_rows_for_label

            # Add a horizontal line below each group of images
            if label < self.dataset.num_classes - 1:
                line_row = current_row
                ax = fig.add_subplot(gs[line_row, :])
                #ax.axhline(y=0, color='black', linewidth=2)
                ax.axis('off')  # Hide this subplot
                current_row += 1  # Move to the next row after the line

        if save_path:
            os.makedirs(save_path, exist_ok=True)
            plt.savefig(os.path.join(save_path, 'cf_vs_org.svg'), format='svg', bbox_inches='tight')
        else:
            plt.show()





        
        
    