import pandas as pd
import numpy as np
from PyEMD import EMD

# 🔹 Step 1: Load Preprocessed Dataset
file_path = "data\complete_data.csv"  # Update with actual path
df = pd.read_csv(file_path)

# Convert timestamp to datetime format
df["cet_cest_timestamp"] = pd.to_datetime(df["cet_cest_timestamp"])

# Select relevant columns
features = ["GR_temperature", "GR_radiation_diffuse_horizontal", "GR_radiation_direct_horizontal", "GR_solar_generation_actual"]
df_selected = df[features]

# 🔹 Step 2: Apply EMD to Extract IMFs
solar_power_series = df_selected["GR_solar_generation_actual"].values
emd = EMD()
IMFs = emd(solar_power_series)  # Extract all IMFs

# Convert all IMFs into DataFrame
df_IMFs = pd.DataFrame(IMFs.T, columns=[f"IMF_{i+1}" for i in range(IMFs.shape[0])])  # Use all IMFs

# 🔹 Step 3: Prepare Historical Values (Fix Shift Direction)
df["T(t-24)"] = df["GR_temperature"].shift(-24)  # Move 24 hours back
df["D_f(t-1)"] = df["GR_radiation_diffuse_horizontal"].shift(-1)  # Move 1 hour back
df["D_r(t-24)"] = df["GR_radiation_direct_horizontal"].shift(-24)  # Move 24 hours back
df["P(t-24)"] = df["GR_solar_generation_actual"].shift(-24)  # Move 24 hours back

# 🔹 Step 4: Construct the Final Dataset
df_final = pd.concat([df_IMFs, df[["T(t-24)", "D_f(t-1)", "D_r(t-24)", "P(t-24)", "GR_solar_generation_actual"]]], axis=1)

# Remove NaN rows caused by shifting
df_final.dropna(inplace=True)

# Rename target variable
df_final.rename(columns={"GR_solar_generation_actual": "Target"}, inplace=True)

# Save dataset for ANFIS training
df_final.to_csv("hybrid_emd_anfis_data.csv", index=False)

print("✅ Dataset ready for training! Data saved as 'hybrid_emd_anfis_data.csv'.")
