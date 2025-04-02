from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np

def apply_minmax_normalization(input_path, output_path):
    # Load the dataset
    df = pd.read_csv(input_path)
    
    # Identify numerical columns to normalize (excluding timestamp column)
    numerical_columns = df.select_dtypes(include=['number']).columns.tolist()
    
    # Apply MinMax scaling
    scaler = MinMaxScaler()
    df[numerical_columns] = scaler.fit_transform(df[numerical_columns])
    
    # Save the normalized dataset
    df.to_csv(output_path, index=False)
    
    print(f"Normalized data saved to {output_path}")

input = r"D:\tuke\summer_semester_24_25\fuzzy_systemy\zapocet\data\complete_data_cleaned.csv"
output = r"D:\tuke\summer_semester_24_25\fuzzy_systemy\zapocet\data\complete_data_cleaned_norm.csv"

apply_minmax_normalization(input, output)