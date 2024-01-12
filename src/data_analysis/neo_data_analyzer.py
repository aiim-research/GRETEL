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
    def create_do_pair_table(cls, do_pair_folder_path, stats_folder_path):
        """This method receives a do-pair folder path. This folder is associated to an specific 
        dataset and oracle combination and should contain folders for each of the explainers tested 
        on that do-pair"""
        results_file_paths = cls.get_json_file_paths(do_pair_folder_path)

        dataset_name = None
        oracle_name = None

        for results_file_uri in results_file_paths:
            with open(results_file_uri, 'r') as results_file_reader:
                results_plain_text = results_file_reader.read()
                results_dict = jsonpickle.decode(results_plain_text)

                if not dataset_name or not oracle_name:
                    dataset_name = results_dict['config']['dataset']['name']
                    oracle_name = results_dict['config']['oracle']['name']

                if dataset_name != results_dict['config']['dataset']['name'] or oracle_name != results_dict['config']['oracle']['name']:
                    raise ValueError('Results files for different do-pairs are contained in the folder')
                
                metrics = [k for k in results_dict.keys if k != 'config']


        raise NotImplementedError()