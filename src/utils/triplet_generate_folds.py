import argparse
import os
import json
import copy


def update_fold_id_for_explainers(data, new_fold_id, target='explainers'):
    if 'parameters' in data.get('experiment', {}):
        propagate_list = data['experiment']['parameters'].get('propagate', [])
        for propagate_dict in propagate_list:
            if target in propagate_dict.get('in_sections', []):
                propagate_dict['params']['fold_id'] = new_fold_id


def perform_ablation(input_data, ablation):

    files_to_return = []
    output_paths = []
    datasets = []
    dopairs = []
    if ablation['dataset'] == 'tree':
        cf = os.path.join("config","JMLR","snippets","datasets","TCR-500-28-7.json")
        ## Create the new datasets, necessary to include in the do-pairs snippets
        with open(cf) as json_file:
            data = json.load(json_file)
            for value in ablation['values']:
                newdata = data.copy()
                par = newdata['parameters']['generator']['parameters']
                newdata['parameters']['generator']['parameters'][ablation['parameter']] = value
                new_file = os.path.join("config","JMLR","snippets","datasets",f"TCR-{par['num_instances']}-{par['num_nodes_per_instance']}-{par['ratio_nodes_in_cycles']}.json")

                datasets.append(new_file)
                with open(new_file, 'w') as outfile:
                    json.dump(newdata, outfile, indent=2)

        ## Edit the new do-pairs snippets
        cf = os.path.join("config","JMLR","snippets","do-pairs","TCR-500-28-7_custom.json")
        with open(cf) as json_file:
            data = json.load(json_file)
            for ds in datasets:
                data['dataset']['compose_gcn'] = ds

                new_file = os.path.join("config","JMLR","snippets","do-pairs",f"{ds.split('/')[-1].replace('.json','')}_custom.json")
                dopairs.append(new_file)
                with open(new_file, 'w') as outfile:
                    json.dump(data, outfile, indent=2)

        ## Edit the configuration files
        for ds in dopairs:
            new_data = input_data.copy()
            for item in new_data['do-pairs']:
                #for each do-pair, replace the value of the do-pair snippet
                item[list(item.keys())[0]] = ds


            #remove node 'ablation_expand' from the new configuration file
            new_data['experiment']['parameters'].pop('ablation_expand', None)

            output_paths.append(ds)
            files_to_return.append(copy.deepcopy(new_data))


    return files_to_return, output_paths

def generate_folds(cf, output_folder, isfolder=False):

    os.makedirs(output_folder, exist_ok=True)

    with open(cf) as json_file:
        data = json.load(json_file)

    triplets = data['doe-triplets']
    # dopairs = data['do-pairs']
    # explainers = data['explainers']
    folds = 10
    targets = data['experiment']['parameters']['expand']['folds']

    cf_name = cf.split(os.sep)[-1].split('.')[-2].split('/')[-1]

    output_folder = output_folder if not isfolder else os.path.join(output_folder, cf_name)


    for tri_num, triplet in enumerate(triplets):
        for fold in range(folds):
            new_data = data.copy()

            # Creating one config file for triplet and fold
            new_data['doe-triplets'] = [triplet]

            for target in targets:
                update_fold_id_for_explainers(new_data, fold, target=target)

            ablation_list = []
            if 'ablation_expand' in new_data['experiment']['parameters']:
                ablation_list = new_data['experiment']['parameters']['ablation_expand']

                
            for ablation in ablation_list:
                ablation = new_data['experiment']['parameters']['ablation_expand']
                files, output_paths = perform_ablation(new_data, ablation)
                
                for newf, op in zip(files, output_paths):

                    filename = op.split(os.sep)[-1]

                    op = os.path.join(os.path.join(output_folder,'temp'))

                    os.makedirs(op, exist_ok=True)

                    op = os.path.join(op, filename)
                    
                    with open(op, 'w') as outfile:
                        json.dump(newf, outfile, indent=2)

                for file in os.listdir(os.path.join(output_folder,'temp')):
                    op = os.path.join(output_folder, file.split('.')[-2])
                    generate_folds(os.path.join(output_folder,'temp',file), op, isfolder=False)

                os.system(f"rm -rf {os.path.join(output_folder,'temp')}")

                return 0

            if len(triplets) == 1:
                print("HERE")
                save_dir = os.path.join(output_folder)
            else:
                save_dir = os.path.join(output_folder, f"{cf_name}_t{tri_num}")

            os.makedirs(save_dir, exist_ok=True)

            new_file = os.path.join(save_dir, f"{cf_name}_t{tri_num}_fold{fold}.json")


            with open(new_file, 'w') as outfile:
                json.dump(new_data, outfile, indent=2)

def main():
    parser = argparse.ArgumentParser(description='Explode the configuration file into a certain number of folds')
    parser.add_argument('--config_file', type=str, help='config file')
    parser.add_argument('--output_folder', type=str, help='output folder')

    args = parser.parse_args()
    cf = args.config_file
    output_folder = args.output_folder if args.output_folder else "."
    isfolder = True if os.path.isdir(cf) else False
    if isfolder:
        config_files = []
        for r, _, f in os.walk(cf):
            for file in f:
                if file.endswith('.json'):
                    config_files.append(os.path.join(r, file))

    else:
        config_files = [cf]


    for cf in config_files:
        generate_folds(cf, output_folder, isfolder=isfolder)        



if __name__ == "__main__":    
    main()


