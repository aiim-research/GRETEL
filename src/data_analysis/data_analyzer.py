from matplotlib.pyplot import table
from src.dataset.dataset_factory import DatasetFactory

import os
import jsonpickle
import numpy as np
from sklearn import metrics
import pandas as pd
import networkx as nx

import matplotlib.pyplot as plt
import statistics
import sys

from src.core.factory_base import get_instance_kvargs

class DataAnalyzer():

    @classmethod
    def get_json_file_paths(cls, folder_path):
        """Given a folder return a list containing the file paths of all json files inside the folder
          or its subfolders"""
        result = []

        for root, dirs, files in os.walk(folder_path):
            for file in files:
                if file.endswith(".json"):
                    result.append(os.path.join(root, file))

        return result
    
    @classmethod
    def create_aggregated_dataframe(cls, results_folder_path):
        """This method receives a do-pair folder path. This folder is associated to an specific 
        dataset and oracle combination and should contain folders for each of the explainers tested 
        on that do-pair"""
        results_file_paths = cls.get_json_file_paths(results_folder_path)

        mega_table = None
        rows = []
        column_names = ['scope', 'dataset', 'oracle', 'explainer', 'fold_id', 'run_id']
        
        for results_file_uri in results_file_paths:
            with open(results_file_uri, 'r') as results_file_reader:
                results_plain_text = results_file_reader.read()
                results_dict = jsonpickle.decode(results_plain_text)

                hashed_scope = results_dict['config']['scope']
                fold_id = results_dict['config']['fold_id']
                run_id = results_dict['config']['run_id']
                extra_columns = results_dict['config']['experiment']['extra_columns']

                # Getting the dataset, oracle and explainer names
                hashed_dataset_name = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(results_file_uri))))
                hashed_oracle_name = os.path.basename(os.path.dirname(os.path.dirname(results_file_uri)))
                hashed_explainer_name = os.path.basename(os.path.dirname(results_file_uri))

                row = [hashed_scope, hashed_dataset_name, hashed_oracle_name, hashed_explainer_name, fold_id, run_id]

                for m_class, m_value in results_dict['results'].items():
                    if len(rows) < 1: # The metric names are only needed the first time
                         column_names.append(m_class)

                    metric = get_instance_kvargs(kls=m_class, param={})
                    agg_values, agg_std = metric.aggregate(m_value)

                    # agg_values = np.mean(m_value)
                    row.append(agg_values)

                rows.append(row)

        mega_table = pd.DataFrame(data=rows, columns=column_names)
        return mega_table


    @classmethod
    def create_aggregated_dataframe_oldstyle(cls, results_folder_path):
        """This method receives a do-pair folder path. This folder is associated to an specific 
        dataset and oracle combination and should contain folders for each of the explainers tested 
        on that do-pair"""
        results_file_paths = cls.get_json_file_paths(results_folder_path)

        mega_dict = {}
        rows = []
        first_iteration = True
        metric_names = []
        
        # Reading the files and creating a dictionaries with aggregated results for each run
        for results_file_uri in results_file_paths:
            with open(results_file_uri, 'r') as results_file_reader:
                results_plain_text = results_file_reader.read()
                results_dict = jsonpickle.decode(results_plain_text)

                # Getting the dataset, oracle and explainer names
                hashed_scope = results_dict['config']['scope']
                hashed_dataset_name = results_dict['hash_ids']['dataset']
                hashed_oracle_name = results_dict['hash_ids']['oracle']
                exp_name = results_dict['hash_ids']['explainer'].split(sep='-')[0]
                hashed_explainer_name = exp_name
                # hashed_explainer_name = results_dict['hash_ids']['explainer']

                # Creating all the necesary levels in the dictionary
                if not hashed_scope in mega_dict:
                    mega_dict[hashed_scope] = {}

                if not hashed_dataset_name in mega_dict[hashed_scope]:
                    mega_dict[hashed_scope][hashed_dataset_name] = {}

                if not hashed_oracle_name in mega_dict[hashed_scope][hashed_dataset_name]:
                    mega_dict[hashed_scope][hashed_dataset_name][hashed_oracle_name] = {}

                if not hashed_explainer_name in mega_dict[hashed_scope][hashed_dataset_name][hashed_oracle_name]:
                    mega_dict[hashed_scope][hashed_dataset_name][hashed_oracle_name][hashed_explainer_name] = []


                correctness_cls, correctness_name = cls.resolve_correctness_class_and_name(results_dict)

                aggregated_metrics = []
                for m_class, m_value in results_dict['results'].items():
                    metric_name = m_class.split('.')[-1]
                    if first_iteration: # The metric names are only needed the first time
                         metric_names.append(metric_name)

                     # Ignoring instances with correctness 0
                    if  m_class != correctness_cls and metric_name != 'FidelityMetric':
                        vals = [x['value'] for x in m_value]
                        correctness_vals = [x['value'] for x in results_dict['results'][correctness_cls]]
                        v_filtered = [item for item, flag in zip(vals, correctness_vals) if flag == 1]
                        
                        metric = get_instance_kvargs(kls=m_class, param={})
                        agg_values, agg_std = metric.aggregate(v_filtered)
                        aggregated_metrics.append(agg_values)
                    else:
                        metric = get_instance_kvargs(kls=m_class, param={})
                        vals = [x['value'] for x in m_value]
                        agg_values, agg_std = metric.aggregate(vals)
                        aggregated_metrics.append(agg_values)

                mega_dict[hashed_scope][hashed_dataset_name][hashed_oracle_name][hashed_explainer_name].append(aggregated_metrics)

            first_iteration = False

        # Creating the header of the table
        column_names = ['scope', 'dataset', 'oracle', 'explainer']
        for m_name in metric_names:
            column_names.append(m_name)
            column_names.append(m_name + '-std')

        # Iterating over the dictionary and agregating different runs and folds together
        rows = []
        for scope_name, datasets in mega_dict.items():
            for dataset_name, oracles in datasets.items():
                for oracle_name, explainers in oracles.items():
                    for explainer_name, runs in explainers.items():
                        row = [scope_name, dataset_name, oracle_name, explainer_name]

                        for m in range(len(metric_names)):
                            m_values = [runs[i][m] for i in range(len(runs))]
                            v_mean = np.mean(m_values)
                            v_std = np.std(m_values)
                            row.append(v_mean)
                            row.append(v_std)

                        rows.append(row)

        # Building the dataframe                  
        result = pd.DataFrame(data=rows, columns=column_names)
        return result


    @classmethod
    def resolve_correctness_class_and_name(cls, results_dict):
        for k in results_dict['results'].keys():
            if 'Correctness' in k or 'correctness' in k:
                return k, k.split('.')[-1]


       