from matplotlib.pyplot import table
from src.dataset.dataset_factory import DatasetFactory

import os
import jsonpickle
import numpy as np
from sklearn import metrics
import pandas as pd
import networkx as nx

import networkx as nx
import matplotlib.pyplot as plt
import statistics
import sys


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
    def create_aggregated_results_dict(cls, results_folder_path):
        """This method receives a do-pair folder path. This folder is associated to an specific 
        dataset and oracle combination and should contain folders for each of the explainers tested 
        on that do-pair"""
        results_file_paths = cls.get_json_file_paths(results_folder_path)

        results_dict = {}
        for results_file_uri in results_file_paths:
            with open(results_file_uri, 'r') as results_file_reader:
                results_plain_text = results_file_reader.read()
                results_dict = jsonpickle.decode(results_plain_text)

                # Getting the dataset and oracle name
                dataset_class = results_dict['config']['dataset']['parameters']['generator']['class']
                dataset_name = dataset_class.split('.')[-1]
                hashed_dataset_name = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(results_file_uri))))

                oracle_class = results_dict['config']['oracle']['class']
                oracle_name = oracle_class.split('.')[-1]
                hashed_oracle_name = os.path.basename(os.path.dirname(os.path.dirname(results_file_uri)))

                # Creating the entries for the dataset and dictionaries if they don't exist
                if dataset_name not in results_dict:
                    results_dict[dataset_name] = {}

                if oracle_name not in results_dict[dataset_name]:
                    results_dict[dataset_name][oracle_name] = {}

                if dataset_name != results_dict['config']['dataset']['name'] or oracle_name != results_dict['config']['oracle']['name']:
                    raise ValueError('Results files for different do-pairs are contained in the folder')
                
                metrics = [k for k in results_dict.keys if k != 'config']


        raise NotImplementedError()