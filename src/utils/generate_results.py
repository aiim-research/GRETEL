import sys
import os
import argparse
import shutil
import time

def clean_up(folder_path, evaluated_path=os.path.join("evaluated")):

    #add timestamp to the evaluated folder
    evaluated_path = os.path.join(evaluated_path, "evaluated_" + str(int(time.time())))
    if not os.path.exists(evaluated_path):
        os.makedirs(evaluated_path)
    
    
    for file_name in os.listdir(folder_path):
        shutil.move(os.path.join(folder_path, file_name), evaluated_path)


current_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(current_dir)
src_dir = os.path.dirname(parent_dir)

sys.path.append(src_dir)

from src.data_analysis.data_analyzer import DataAnalyzer as dan

parser = argparse.ArgumentParser(description='Analyze results and generate statistics')
parser.add_argument('--output_file', type=str, help='output file')
parser.add_argument('--source_folder', type=str, help='source folder')

args = parser.parse_args()

os.chdir(src_dir)

results_path = os.path.join(src_dir, args.source_folder)
stats_file_path = os.path.join(src_dir, args.output_file)
res = dan.create_aggregated_dataframe(results_path)
#clean_up(results_path)
res.to_csv(stats_file_path)