import argparse
import os
import json

def make_dir_if_not_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)

def update_fold_id_for_explainers(data, new_fold_id, target='explainers'):
    if 'parameters' in data.get('experiment', {}):
        propagate_list = data['experiment']['parameters'].get('propagate', [])
        for propagate_dict in propagate_list:
            if target in propagate_dict.get('in_sections', []):
                propagate_dict['params']['fold_id'] = new_fold_id

def generate_folds(cf, output_folder, isfolder=False):

    make_dir_if_not_exists(output_folder)

    with open(cf) as json_file:
        data = json.load(json_file)

    dopairs = data['do-pairs']
    explainers = data['explainers']
    folds = 10
    targets = data['experiment']['parameters']['expand']['folds']

    cf_name = cf.split(os.sep)[-1].split('.')[-2].split('/')[-1]

    output_folder = output_folder if not isfolder else os.path.join(output_folder, cf_name)

    for number_do_pair, do_pair in enumerate(dopairs):
        for number_explainer, explainer in enumerate(explainers):
            for fold in range(folds):
                new_data = data.copy()
                new_data['do-pairs'] = [do_pair]
                new_data['explainers'] = [explainer]

                #do_pair_name = list(do_pair.values())[0].split("_")[-1].split('.')[-2]
                #explainer_name = list(explainer.values())[0].split("_")[-1].split('.')[-1]

                for target in targets:
                    update_fold_id_for_explainers(new_data, fold, target=target)

                if len(dopairs) == 1:
                    save_dir = os.path.join(output_folder)
                else:
                    save_dir = os.path.join(output_folder, f"{cf_name}_do{number_do_pair}_e{number_explainer}")

                make_dir_if_not_exists(save_dir)

                new_file = os.path.join(save_dir, f"{cf_name}_do{number_do_pair}_e{number_explainer}_f{fold}.json")


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


