import pandas as pd
import sys

def process_csv_files(file_names):
    # Initialize an empty DataFrame to hold the aggregated data
    aggregated_data = pd.DataFrame()

    for file_name in file_names:
        # Load each CSV file
        data = pd.read_csv(file_name)

        # Preprocess the data
        data['Correctness'] = data['Correctness']  # Keep Correctness as is
        
        # Clean up explainer names: Remove 'GenerateMinimize', 'Explainer', and parenthesis
        data['explainer'] = (
            data['explainer']
            .str.replace('GenerateMinimize', '', regex=False)
            .str.replace('Explainer', '', regex=False)
            .str.replace(r'\(|\)', '', regex=True)
            .str.strip()
        )

        # Group by explainer and compute average values for the required fields
        data_grouped = data.groupby('explainer').agg({
            'GraphEditDistance': 'mean',
            'FeatureEditDistance': 'mean',
            'Correctness': 'mean',
            'OracleCalls': 'mean'
        }).reset_index()

        # Add to the aggregated DataFrame
        if aggregated_data.empty:
            aggregated_data = data_grouped
        else:
            # Merge and average the values of explainers with the same name
            aggregated_data = pd.concat([aggregated_data, data_grouped]).groupby('explainer', as_index=False).mean()

    # Sort the aggregated data alphabetically by explainer name
    aggregated_data = aggregated_data.sort_values('explainer')

    # Save the table to a CSV file with headers
    aggregated_data.to_csv('aggregated_table.csv', index=False, header=True)
    
    print("Aggregated table has been saved as 'aggregated_table.csv'.")
    print(aggregated_data)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python script.py <file1.csv> <file2.csv> ...")
    else:
        file_names = sys.argv[1:]
        process_csv_files(file_names)
