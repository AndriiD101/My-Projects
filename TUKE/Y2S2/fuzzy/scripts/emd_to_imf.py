import pandas as pd
import numpy as np
from PyEMD import EMD
import matplotlib.pyplot as plt

# === Load preprocessed normalized data ===
df = pd.read_csv(r"C:\Users\denys\Desktop\My-Projects\TUKE\Y2S2\fuzzy\data\complete_data.csv")
solar_signal = df["GR_solar_generation_actual"].values

# === Apply Empirical Mode Decomposition ===
emd = EMD()
imfs = emd(solar_signal)  # shape: [n_imfs, n_samples]

# === Plot IMFs (optional visualization) ===
plt.figure(figsize=(12, 2 * (imfs.shape[0] + 1)))
plt.subplot(imfs.shape[0] + 1, 1, 1)
plt.plot(solar_signal)
plt.title("Original Solar Signal")

for i, imf in enumerate(imfs):
    plt.subplot(imfs.shape[0] + 1, 1, i + 2)
    plt.plot(imf)
    plt.title(f"IMF {i + 1}")

plt.tight_layout()
plt.show()

# === Save IMFs ===
# Method 1: Save as .npy (best for reuse)
np.save("imfs.npy", imfs)

# Method 2: Also save as .csv (for inspection/debug)
pd.DataFrame(imfs.T, columns=[f"IMF_{i+1}" for i in range(imfs.shape[0])]).to_csv("imfs.csv", index=False)

print(f"✅ Extracted {imfs.shape[0]} IMFs and saved to 'imfs.npy' and 'imfs.csv'")
