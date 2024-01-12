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

    def __init__(self) -> None:
        super.__init__()

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
        result_file_paths = cls.get_json_file_paths(do_pair_folder_path)

        raise NotImplementedError()