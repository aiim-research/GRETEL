import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

output = os.path.join("visualizations", "methods_comparison_tc28_gcn.png")

results_path = os.path.join("output","processed_results","tcr_gcn.csv")
results_file = pd.read_csv(results_path, index_col=0)
old_results = pd.read_csv( os.path.join("output/old_results_tcr.csv"), index_col=0 )

results_file['dataset'] = results_file['dataset'].str.split("-").str[0]
results_file['oracle'] = results_file['oracle'].str.split("-").str[0]
results_file = results_file.loc[:,~results_file.columns.str.endswith("-std")]

old_results = old_results.loc[:,~old_results.columns.str.endswith("-std")]
old_results['dataset'] = old_results['dataset'].str.split("-").str[0]
old_results['oracle'] = old_results['oracle'].str.split("-").str[0]

exclude = ['RuntimeMetric', 'OracleAccuracyMetric', 'OracleCallsMetric']

df = results_file
prev_df = old_results

dataset_oracle_pairs = df.groupby(['dataset', 'oracle']).size().reset_index()[['dataset', 'oracle']]


metrics = [col for col in df.columns if 'Metric' in col and '-std' not in col and col not in exclude]

# Create figure with subplots
fig, axes = plt.subplots(nrows=len(dataset_oracle_pairs), ncols=len(metrics), figsize=(20, 10), squeeze=False)

fig.suptitle('Metrics Comparison by Dataset and Oracle', fontsize=16, fontweight='bold')

for i, (idx, row) in enumerate(dataset_oracle_pairs.iterrows()):
    for j, metric in enumerate(metrics):
        ax = axes[i][j]
        subset = df[(df['dataset'] == row['dataset']) & (df['oracle'] == row['oracle'])]

        prev_subset = prev_df[(prev_df['dataset'] == row['dataset'])]
        #error = subset[f"{metric}-std"].values
        pivoted = subset.pivot(index='explainer', columns='dataset', values=metric)
        prev_pivoted = prev_subset.pivot(index='explainer', columns='dataset', values=metric)
        colors = ['orange' if explainer == 'RSGG' else 'grey' for explainer in pivoted.index]
        
        bar_plot = pivoted.plot(kind='bar', ax=ax, capsize=4, color=colors, legend=None)
        ax.set_title(metric.replace('Metric', ''))
        
        # Overlay markers from previous dataset
        #if not prev_pivoted.empty:
        #    for k, explainer in enumerate(pivoted.index):
        #        ax.scatter(x=k, y=prev_pivoted.loc[explainer], color='black', marker='o', zorder=5)

        ax.set_xlabel('')

    axes[i, 0].text(-0.3, 1.1, f'Dataset: {row["dataset"]} - Oracle: {row["oracle"]}',
                    fontsize=14, transform=axes[i, 0].transAxes, verticalalignment='top')

plt.tight_layout()
plt.subplots_adjust(top=0.9)
plt.savefig(output, bbox_inches='tight')

