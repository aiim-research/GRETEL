from collections import OrderedDict
import copy
import json
import os
import torch
import numpy as np

def update_saved_pyg(input_file,output_file):
    old_model =  torch.load(input_file, map_location=torch.device('cpu'))
    fixed_model = OrderedDict([(k.replace("grpah", "graph"), v) if 'grpah' in k else (k, v) for k, v in old_model.items()])
    torch.save(fixed_model,output_file)

def sanitize_dir_pyg(based_dir,prefix,model_name='explainer'):
    for file in os.listdir(based_dir):
        if file.startswith(prefix):            
            model_file_name = os.path.join(based_dir,file,model_name)
            if os.path.exists(model_file_name):
                old_file_name = os.path.join(based_dir,file,"OLD_"+model_name)

                print("Sanitizing: "+model_file_name)

                os.rename(model_file_name, old_file_name)
                print("Renamed to: "+old_file_name)

                update_saved_pyg(old_file_name,model_file_name)
                print("Complete")

def unfold_confs(based_dir,out_dir,prefix,num_folds=10):
    for dir in os.listdir(based_dir):
        if dir.startswith(prefix) and os.path.isdir(os.path.join(based_dir,dir)):
            ''' if not os.path.exists(os.path.join(out_dir,dir)):
                os.makedirs(os.path.join(out_dir,dir)) '''
            for sub_dir in os.listdir(os.path.join(based_dir,dir)):
                if os.path.isdir(os.path.join(based_dir,dir,sub_dir)):
                    if not os.path.exists(os.path.join(out_dir,dir,sub_dir)):
                        os.makedirs(os.path.join(out_dir,dir,sub_dir))
                    print("Processing subfolder: "+os.path.join(based_dir,dir,sub_dir))
                    for conf_file in os.listdir(os.path.join(based_dir,dir,sub_dir)):
                        #print(conf_file)
                        in_file = os.path.join(based_dir,dir,sub_dir,conf_file)
                        out_file = os.path.join(out_dir,dir,sub_dir,conf_file)

                        with open(in_file, 'r') as config_reader:
                            configuration = json.load(config_reader)                                                    
                            for fold_id in range(num_folds):
                                current_conf =  copy.deepcopy(configuration)
                                for exp  in current_conf['explainers']:
                                    exp['parameters']['fold_id']=fold_id
                                
                                out_file = os.path.join(out_dir,dir,sub_dir,conf_file[:-5]+'_'+str(fold_id)+'.json')
                                with open(out_file, 'w') as o_file:
                                    json.dump(current_conf, o_file)
                                print(out_file)
                                
def pad_adj_matrix(adj_matrix, target_dimension):
    # Get the current dimensions of the adjacency matrix
    current_rows, current_cols = adj_matrix.shape
    # Calculate the amount of padding needed for rows and columns
    pad_rows = max(0, target_dimension - current_rows)
    pad_cols = max(0, target_dimension - current_cols)
    # Pad the adjacency matrix with zeros
    return np.pad(adj_matrix, ((0, pad_rows), (0, pad_cols)), mode='constant')     

# from src.utils.utils import update_saved_pyg 

#input_file="/NFSHOME/mprado/CODE/gretel-steel-2/GRETEL/data/explainers/clear_fit_on_tree-cycles_instances-500_nodes_per_inst-28_nodes_in_cycles-7_fold_id=0_batch_size_ratio=0.15_alpha=0.4_lr=0.01_weight_decay=5e-05_epochs=600_dropout=0.1/old_explainer"
#output_file="/NFSHOME/mprado/CODE/gretel-steel-2/GRETEL/data/explainers/clear_fit_on_tree-cycles_instances-500_nodes_per_inst-28_nodes_in_cycles-7_fold_id=0_batch_size_ratio=0.15_alpha=0.4_lr=0.01_weight_decay=5e-05_epochs=600_dropout=0.1/explainer"
#update_saved_pyg(input_file,output_file)


#based_dir='/NFSHOME/mprado/CODE/gretel-steel-2/GRETEL/data/explainers/'
#sanitize_dir_pyg(based_dir,"clear")
#unfold_confs("config/aaai","AAAI/config","ablation")
