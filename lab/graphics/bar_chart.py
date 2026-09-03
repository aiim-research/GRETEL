import pandas as pd
import matplotlib.pyplot as plt
import sys

def generate_charts(file_name):
    # Load the CSV file
    data = pd.read_csv(file_name)

    # Preprocess the data
    data['Failure'] = 1 - data['Correctness']  # Invert Correctness to Failure
    
    # Clean up explainer names: Remove 'GenerateMinimize', 'Explainer', and parenthesis
    data['explainer'] = (
        data['explainer']
        .str.replace('GenerateMinimize', '', regex=False)
        .str.replace('Explainer', '', regex=False)
        .str.replace(r'\(|\)', '', regex=True)
        .str.strip()
    )
    
    # Sort alphabetically by explainer name
    data_sorted = data.sort_values('explainer')

    # Extract the necessary data
    explainers = data_sorted['explainer']
    graph_edit_distance = data_sorted['GraphEditDistance']
    feature_edit_distance = data_sorted['FeatureEditDistance']
    failure = data_sorted['Failure']
    oracle_calls = data_sorted['OracleCalls']

    # Create the bar charts
    fig, axs = plt.subplots(2, 2, figsize=(16, 12))
    bar_width = 0.5

    # Reduce font size for better fit
    plt.rc('font', size=10)  # Reduce the overall font size

    # Plot each chart
    charts = [
        ('Graph Edit Distance', graph_edit_distance, axs[0, 0]),
        ('Feature Edit Distance', feature_edit_distance, axs[0, 1]),
        ('Failure', failure, axs[1, 0]),
        ('Oracle Calls', oracle_calls, axs[1, 1]),
    ]

    for title, values, ax in charts:
        ax.bar(explainers, values, width=bar_width, color='skyblue', edgecolor='black')
        ax.set_title(title, fontsize=12)  # Reduce title font size slightly
        ax.set_xlabel('Explainer', fontsize=10)
        ax.set_ylabel('Value', fontsize=10)
        ax.set_xticks(range(len(explainers)))
        ax.set_xticklabels(explainers, rotation=45, ha='right')

        # Annotate bars with exact values
        for idx, value in enumerate(values):
            ax.text(idx, value + max(values) * 0.01, f'{value:.2f}', ha='center', va='bottom', fontsize=8)

    # Adjust layout and save the plot
    plt.tight_layout()
    plt.savefig('output_charts.png')  # Save the chart as a file
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python script.py <file_name>")
    else:
        file_name = sys.argv[1]
        generate_charts(file_name)
