import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

#* Task 1 - Load and Explore the Dataset

# Load the Iris dataset from sklearn
iris = load_iris()
iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
iris_df['species'] = iris.target_names[iris.target]  # Add species column

# Display the first 5 rows
print(iris_df.head())

# Check data types
print(iris_df.info())

# Check for missing values
print(iris_df.isnull().sum())


#* Task 2 - Basic Data Analysis

print(iris_df.describe())

# Group by Species and Compute Mean
species_mean = iris_df.groupby('species').mean()
print(species_mean)

#* Task 3 - Data Visualization

# Line Chart (Trend of Sepal Length Across Samples)
plt.figure(figsize=(10, 4))
plt.plot(iris_df['sepal length (cm)'], label='Sepal Length')
plt.title("Sepal Length Trend Across Iris Samples")
plt.xlabel("Sample Index")
plt.ylabel("Sepal Length (cm)")
plt.legend()
plt.show()

# Bar Chart (Average Petal Length by Species)
plt.figure(figsize=(8, 5))
sns.barplot(x='species', y='petal length (cm)', data=iris_df, estimator=np.mean)
plt.title("Average Petal Length by Species")
plt.xlabel("Species")
plt.ylabel("Petal Length (cm)")
plt.show()

# Histogram (Distribution of Sepal Width)
plt.figure(figsize=(8, 5))
sns.histplot(iris_df['sepal width (cm)'], bins=15, kde=True)
plt.title("Distribution of Sepal Width")
plt.xlabel("Sepal Width (cm)")
plt.ylabel("Frequency")
plt.show()

# Scatter Plot (Sepal Length vs. Petal Length)
plt.figure(figsize=(8, 5))
sns.scatterplot(x='sepal length (cm)', y='petal length (cm)', hue='species', data=iris_df)
plt.title("Sepal Length vs. Petal Length")
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Petal Length (cm)")
plt.legend()
plt.show()