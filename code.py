# Import required libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('side_effect_data 1.xlsx')

# 1. Basic Information about the dataset
print("Dataset Info:")
df.info()

# 2. Statistical Summary of the dataset
print("\nStatistical Summary:")
print(df.describe())

# 3. Checking for Missing Values
print("\nMissing Values:")
print(df.isnull().sum())

# 4. Checking for Duplicates
print("\nDuplicate Rows:")
print(df.duplicated().sum())

# 5. Data Types of Columns
print("\nData Types:")
print(df.dtypes)

# 6. Distribution of Numeric Variables (Histograms)
df.hist(bins=20, figsize=(15, 10))
plt.suptitle("Distribution of Numeric Features", fontsize=16)
plt.show()

# 7. Scatter Plot Matrix (Numeric Features) to check relationships
sns.pairplot(df)
plt.suptitle("Scatter Plot Matrix of Numeric Features", fontsize=16)
plt.show()

# 8. Correlation Heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', linewidths=0.5)
plt.title("Correlation Heatmap", fontsize=16)
plt.show()

# 9. Count Plot for Categorical Variables (if any)
categorical_cols = df.select_dtypes(include=['object']).columns
for col in categorical_cols:
    plt.figure(figsize=(8, 6))
    sns.countplot(x=col, data=df)
    plt.title(f"Count Plot of {col}")
    plt.xticks(rotation=45)
    plt.show()

# 10. Boxplots for Detecting Outliers in Numeric Features
numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
for col in numeric_cols:
    plt.figure(figsize=(8, 6))
    sns.boxplot(x=df[col])
    plt.title(f"Boxplot of {col}")
    plt.show()