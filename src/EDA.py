# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
file_path = "data/ckc.csv"  # Replace with actual dataset path
df = pd.read_csv(file_path)

# Standardize column names (convert to lowercase, replace spaces with underscores)
df.columns = df.columns.str.lower().str.replace(" ", "_")

# Display dataset info
print("Dataset Info:")
print(df.info())

# Summary statistics
print("\nSummary Statistics:")
print(df.describe())

# Check for missing values
print("\nMissing Values Count:")
print(df.isnull().sum())

# Visualizing missing values using a heatmap
plt.figure(figsize=(12, 6))
sns.heatmap(df.isnull(), cbar=False, cmap="viridis")
plt.title("Missing Values Heatmap")
plt.show()

# Identify numerical and categorical columns
num_cols = df.select_dtypes(include=['number']).columns
cat_cols = df.select_dtypes(include=['object']).columns

# Visualize feature distributions
df[num_cols].hist(figsize=(20,10 ), bins=20, edgecolor='black')
plt.suptitle("Feature Distributions", fontsize=16)
plt.show()

# Boxplots for outlier detection
plt.figure(figsize=(30, 24))
df[num_cols].boxplot(rot=90)
plt.title("Boxplots for Outlier Detection")
plt.show()

# Correlation heatmap
plt.figure(figsize=(24,38))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", linewidths=0.5)
plt.title("Correlation Heatmap")
plt.show()

# Pairplot for selected important features
selected_features = ["age", "bmi", "serumcreatinine", "gfr", "bunlevels"]  # Modify based on dataset
sns.pairplot(df[selected_features])
plt.show()

print("\nExploratory Data Analysis Completed Successfully!")
