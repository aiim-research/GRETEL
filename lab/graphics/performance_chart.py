import pandas as pd
import matplotlib.pyplot as plt
import sys

def generate_performance_chart(file_name):
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

    # Normalize the values to proportions (0 to 1) for each metric
    data['GraphEditDistance_Proportion'] = data['GraphEditDistance'] / data['GraphEditDistance'].max()
    data['FeatureEditDistance_Proportion'] = data['FeatureEditDistance'] / data['FeatureEditDistance'].max()
    data['Failure_Proportion'] = data['Failure'] / data['Failure'].max()
    data['OracleCalls_Proportion'] = data['OracleCalls'] / data['OracleCalls'].max()

    # Compute performance as the average of proportions
    data['Performance'] = (
        data[['GraphEditDistance_Proportion', 'FeatureEditDistance_Proportion', 
              'Failure_Proportion', 'OracleCalls_Proportion']].mean(axis=1)
    )

    # Sort data alphabetically by explainer name
    data_sorted = data.sort_values('explainer')

    # Extract the explainers and their performance scores
    explainers = data_sorted['explainer']
    performance = data_sorted['Performance']

    # Create the performance bar chart
    plt.figure(figsize=(12, 6))
    plt.bar(explainers, performance, color='skyblue', edgecolor='black')
    plt.title('Performance Score', fontsize=14)
    plt.xlabel('Explainer', fontsize=12)
    plt.ylabel('Performance (Average Proportion)', fontsize=12)
    plt.xticks(rotation=45, ha='right', fontsize=10)

    # Annotate bars with performance values
    for idx, value in enumerate(performance):
        plt.text(idx, value + 0.02, f'{value:.2f}', ha='center', va='bottom', fontsize=10)

    # Adjust layout and save the plot
    plt.tight_layout()
    plt.savefig('performance_chart.png')  # Save the chart as a file
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python script.py <file_name>")
    else:
        file_name = sys.argv[1]
        generate_performance_chart(file_name)
