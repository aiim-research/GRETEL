import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the CSV file
file_path = sys.argv[1]
data = pd.read_csv(file_path)

# Update the explainer labels to only include content within parentheses
data['explainer'] = data['explainer'].str.extract(r'\((.*?)\)')[0]

# Set the bar width and positions
bar_width = 0.25
x = np.arange(len(data.index))

# Create the figure and axes
plt.figure(figsize=(14, 8))

# Plot each metric with a different bar position
bars1 = plt.bar(x - bar_width, data['Runtime'].clip(upper=20), width=bar_width, label='Runtime', color='b')
bars2 = plt.bar(x, data['GraphEditDistance'].clip(upper=20), width=bar_width, label='Graph Edit Distance', color='g')
bars3 = plt.bar(x + bar_width, data['Correctness'].clip(upper=20), width=bar_width, label='Correctness', color='r')

# Adding experiment labels below the bars
plt.xticks(x, data['explainer'], rotation=45, ha='right', fontsize=9)

# Adding axis labels and legend only
plt.xlabel('Explainer')
plt.ylabel('Values')
plt.legend()

# Display exact values on top of each bar with a small font size
for bars, values in zip([bars1, bars2, bars3], [data['Runtime'], data['GraphEditDistance'], data['Correctness']]):
    for bar, value in zip(bars, values):
        height = min(value, 20)  # Set the capped height for display
        plt.text(bar.get_x() + bar.get_width() / 2, height + 0.2,
                 f'{value:.2f}' if value < 20 else f'{value:.2f}↑', 
                 ha='center', va='bottom', fontsize=6, color='black')

# Limit y-axis to emphasize smaller values
plt.ylim(0, 20)

# Adjust layout and display plot
plt.tight_layout()
plt.show()
