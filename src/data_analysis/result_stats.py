from src.data_analysis.data_analyzer import DataAnalyzer

data_store_path = './data/datasets/'
dtan = DataAnalyzer('./output', './stats')

# data_store_path = './data/datasets'
# dtan = DataAnalyzer('./output/tc500-28-7', './stats/tc500-28-7')
# # dtan = DataAnalyzer('C:\\Work\\GNN\\Mine\\GRETEL2\\GRETEL\\output_legacy\\synthetic', 'C:\\Work\\GNN\\Mine\\GRETEL2\\GRETEL\\stats_legacy')

# dtan.aggregate_data()
# dtan.aggregate_runs()
# dtan.create_tables_by_oracle_dataset()

datasets =[
    # {"name": "adhd", "parameters": {} },
    {"name": "autism", "parameters": {} },
    {"name": "tree-cycles", "parameters": {"n_inst": 500, "n_per_inst": 28, "n_in_cycles": 7} },
    # {"name": "tree-infinity", "parameters": {"n_inst": 500, "n_per_inst": 300, "n_infinities": 10, "n_broken_infinities": 10}},
    # {"name": "bbbp", "parameters": {"force_fixed_nodes": False}},
    # {"name": "hiv", "parameters": {"force_fixed_nodes": False}}
]

dtan.get_datasets_stats(datasets, data_store_path)


### dataset_stats.py

'''from src.data_analysis.data_analyzer import DataAnalyzer

# data_store_path = './data/datasets/'
# dtan = DataAnalyzer('./output', './stats')

data_store_path = './data/datasets'
dtan = DataAnalyzer('./output/steel/BBBP', './stats/steel/')


datasets =[
    {"name": "tree-cycles", "parameters": {"n_inst": 500, "n_per_inst": 32, "n_in_cycles": 7}},
    {"name": "tree-cycles", "parameters": {"n_inst": 500, "n_per_inst": 28, "n_in_cycles": 7}},
    {"name": "tree-cycles", "parameters": {"n_inst": 500, "n_per_inst": 48, "n_in_cycles": 7}},
    {"name": "autism", "parameters": {} }
]

dtan.get_datasets_stats(datasets, data_store_path)'''