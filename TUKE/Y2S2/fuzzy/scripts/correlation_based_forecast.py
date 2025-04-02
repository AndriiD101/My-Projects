import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# === CONFIG ===
SELECTED_IMFS = [1, 2, 9, 10, 11]  # 1-based indexing
IN_FEATURES = 4
MF_PER_INPUT = 3
MODEL_PREFIX = "models/anfis_imf"
MODEL_EXT = ".pt"

# === Load and align input features ===
df = pd.read_csv(r"D:\tuke\summer_semester_24_25\fuzzy_systemy\zapocet\data\complete_data.csv")
features = pd.DataFrame({
    "T(t-24)": df["GR_temperature"].shift(-24),
    "Df(t-1)": df["GR_radiation_diffuse_horizontal"].shift(-1),
    "Dr(t-24)": df["GR_radiation_direct_horizontal"].shift(-24),
    "P(t-24)": df["GR_solar_generation_actual"].shift(-24),
})

valid_idx = features.dropna().index
X = features.loc[valid_idx].reset_index(drop=True).values
X_tensor = torch.tensor(X, dtype=torch.float32)

# === Load ground truth
solar_true = df["GR_solar_generation_actual"].iloc[valid_idx].reset_index(drop=True).values

# === Define ANFIS model
class BellMF(nn.Module):
    def __init__(self, in_features, mf_per_input):
        super().__init__()
        self.a = nn.Parameter(torch.ones(in_features, mf_per_input))
        self.b = nn.Parameter(torch.ones(in_features, mf_per_input))
        self.c = nn.Parameter(torch.linspace(0, 1, mf_per_input).repeat(in_features, 1))

    def forward(self, x):
        x_expanded = x.unsqueeze(2)
        return 1 / (1 + torch.abs((x_expanded - self.c.unsqueeze(0)) / self.a.unsqueeze(0)) ** (2 * self.b.unsqueeze(0)))

class SimpleANFIS(nn.Module):
    def __init__(self, in_features, mf_per_input, hidden_dim=16):
        super().__init__()
        self.mf_layer = BellMF(in_features, mf_per_input)
        self.rules = nn.Sequential(
            nn.Linear(in_features * mf_per_input, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        mf_values = self.mf_layer(x)
        combined = mf_values.reshape(x.size(0), -1)
        return self.rules(combined)

# === Forecast reconstruction from selected IMFs
forecast = np.zeros(X.shape[0])

for i in SELECTED_IMFS:
    model = SimpleANFIS(IN_FEATURES, MF_PER_INPUT, hidden_dim=16)
    model.load_state_dict(torch.load(f"{MODEL_PREFIX}{i}{MODEL_EXT}"))
    model.eval()
    with torch.no_grad():
        pred = model(X_tensor).numpy().flatten()
        forecast += pred
    print(f"✅ Added prediction from IMF {i}")

# === Save forecast
pd.Series(forecast).to_csv("solar_forecast_targeted.csv", index=False)

# === Plot forecast vs actual
plt.figure(figsize=(14, 5))
plt.plot(solar_true, label="Actual Solar Generation", alpha=0.7)
plt.plot(forecast, label="Forecasted (Selected IMFs)", alpha=0.7)
plt.title("Forecast vs Actual (Using Selected IMFs)")
plt.xlabel("Time Steps")
plt.ylabel("Normalized Solar Output")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
