import numpy as np
import matplotlib.pyplot as plt
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
        
        originals = dict()
        counterfactuals = dict()
    
        for file in filepaths:
            with open(file, 'r') as f:
                data = decode(f.read())
                original_id = data['original_id']
                
                cf = data['counterfactual_adj']
                original = self.dataset.get_instance(original_id).data
                
                if not data['correctness']:
                    cf = np.array([])
                    
                originals[original_id] = original
                counterfactuals[original_id] = cf
            
        all_changes = list()    
        for id in originals.keys():
            changes = self.__plot_matrix_changes(originals[id], counterfactuals[id])
            all_changes.append(changes)
            
        self.plot_matrices_in_row(all_changes, 
                                  save_path=os.path.join(self.dump_path, self.dataset.name, self.oracle.name, self.explainer.name))
    
    
    def __plot_matrix_changes(self, A, B):
        if not len(B):
            changes_matrix = np.ones((A.shape[0], A.shape[1], 3), dtype=np.uint8)
            changes_matrix *= [255, 255, 255]
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
        num_matrices = len(matrices)

        fig, axes = plt.subplots(1, num_matrices, figsize=(4 * num_matrices, 4))

        for i in range(num_matrices):
            try:
                ax = axes[i] if num_matrices > 1 else axes  # Handle the case with only one matrix
                ax.imshow(matrices[i], vmin=0, vmax=255)
                ax.set_xticks([])  # Hide x-axis ticks
                ax.set_yticks([])  # Hide y-axis 
            except:
                print(f'Skipping {i}-th matrix of changes')
            
        if save_path:
            os.makedirs(save_path, exist_ok=True)
            plt.savefig(os.path.join(save_path, 'cf_vs_org.svg'), format='svg', bbox_inches='tight')
        else:
            plt.show()
        
        
    