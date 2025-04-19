import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.preprocessing import LabelEncoder
import graphviz
from graphviz import Digraph

# Ensure the Graphviz executables are available on the system PATH
import os
os.environ["PATH"] += os.pathsep + r"C:\Program Files\Graphviz\bin"

# Load the comparison data
comparison_df = pd.read_csv('data/model_comparison.csv')

# Set the model names as the index
comparison_df.set_index('Model', inplace=True)

# Define the list of possible columns for metrics
possible_columns = [
    'Accuracy', 'Precision', 'Recall', 'F1 Score',
    'Precision_0', 'Precision_1', 'Recall_0', 'Recall_1', 'F1_Score_0', 'F1_Score_1'
]

# Filter out only the columns that exist in the DataFrame
metrics_to_plot = [col for col in possible_columns if col in comparison_df.columns]

# Plot the metrics for comparison
comparison_df[metrics_to_plot].plot(kind='bar', figsize=(12, 8))

# Add title and labels
plt.title('Model Comparison for Different Metrics', fontsize=16)
plt.ylabel('Score', fontsize=12)
plt.xticks(rotation=0)
plt.ylim(0, 1.1)

# Add a horizontal line at the perfect score (1.0)
plt.axhline(y=1.0, color='r', linestyle='--', label='Perfect Score')

# Add legend to the plot
plt.legend(title="Metrics")

# Ensure the 'evolution' directory exists
evolution_dir = 'evolution'
os.makedirs(evolution_dir, exist_ok=True)

# Save the plot to the 'evolution' directory
plt.savefig(f'{evolution_dir}/model_comparison.png')

# Display the plot
plt.show()

print("Model comparison plot saved successfully.")

# --- Heatmap of the correlation between metrics ---
# Compute the correlation matrix
corr_matrix = comparison_df[metrics_to_plot].corr()

# Plot the heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', cbar=True)
plt.title('Correlation Heatmap of Model Metrics', fontsize=16)
plt.savefig(f'{evolution_dir}/metrics_heatmap.png')
plt.show()

print("Heatmap saved successfully.")

# --- Flowchart (Example of a simple flowchart using graphviz) ---
# Create a flowchart using Graphviz
dot = Digraph(comment='Model Evaluation Pipeline')

dot.node('A', 'Load Data')
dot.node('B', 'Preprocess Data')
dot.node('C', 'Train Models')
dot.node('D', 'Evaluate Models')
dot.node('E', 'Generate Plots')
dot.node('F', 'Save Results')

dot.edges(['AB', 'BC', 'CD', 'DE', 'EF'])

# Render the flowchart as a PNG
flowchart_path = f'{evolution_dir}/model_evaluation_flowchart.png'
dot.render(flowchart_path, format='png', cleanup=True)

print(f"Flowchart saved at: {flowchart_path}")

# --- Pre-information (Summary of the Data or Model Metrics) ---
# Display a summary of the metrics before plotting
summary_info = comparison_df[metrics_to_plot].describe()

# Save summary to a text file
summary_path = f'{evolution_dir}/model_summary.txt'
with open(summary_path, 'w') as f:
    f.write("Model Metrics Summary:\n\n")
    f.write(summary_info.to_string())

print(f"Model summary saved at: {summary_path}")
