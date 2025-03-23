import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# For display purposes
pd.set_option('display.max_columns', None)

# Load the dataset
df = pd.read_csv("bank-full.csv", sep=';')  # Adjust path if needed

# ----- BASIC STATS -----
print("Dataset Shape:", df.shape)
print("\nColumn Data Types:\n", df.dtypes)
print("\nMissing Values:\n", df.isnull().sum())
print("\nStatistical Description (Numerical Columns):\n", df.describe())
print("\nTarget Distribution:\n", df['y'].value_counts())

# ----- HISTOGRAMS FOR NUMERICAL FEATURES -----
numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

plt.figure(figsize=(15, 10))
for i, col in enumerate(numerical_cols):
    plt.subplot(3, 3, i + 1)
    sns.histplot(df[col], kde=True, bins=30, color='skyblue')
    plt.title(f'Distribution of {col}')
    plt.xlabel(col)
    plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("eda_histograms.png")
plt.show()

# ----- BAR PLOTS FOR CATEGORICAL FEATURES -----
categorical_cols = df.select_dtypes(include='object').columns.tolist()
categorical_cols.remove('y')  # Exclude target

plt.figure(figsize=(15, 12))
for i, col in enumerate(categorical_cols[:6]):
    plt.subplot(3, 2, i + 1)
    df[col].value_counts().plot(kind='bar', color='orange')
    plt.title(f'Distribution of {col}')
    plt.xlabel(col)
    plt.ylabel("Count")
    plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("eda_barplots.png")
plt.show()

# ----- CORRELATION MATRIX -----
plt.figure(figsize=(10, 8))
corr = df[numerical_cols].corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", square=True)
plt.title("🔗 Correlation Matrix of Numerical Features")
plt.savefig("eda_correlation_matrix.png")
plt.show()
