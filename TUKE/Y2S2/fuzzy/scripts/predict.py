import torch
import torch.nn as nn
import numpy as np
import pandas as pd

# === CONFIG ===
N_IMFS = 11
MF_PER_INPUT = 3
IN_FEATURES = 4
MODEL_PREFIX = "models/anfis_imf"
MODEL_EXT = ".pt"
ROW_INDEX = 2  # The row we want to forecast (adjust as needed)

# === Load data ===
df = pd.read_csv(r"C:\Users\denys\Desktop\My-Projects\TUKE\Y2S2\fuzzy\data\complete_data.csv")

# === Get the past values (paper recommends T(t-24), Df(t-1), Dr(t-24), P(t-24)) ===
T_24  = df.loc[ROW_INDEX - 24, "GR_temperature"]
Df_1  = df.loc[ROW_INDEX -  1, "GR_radiation_diffuse_horizontal"]
Dr_24 = df.loc[ROW_INDEX - 24, "GR_radiation_direct_horizontal"]
P_24  = df.loc[ROW_INDEX - 24, "GR_solar_generation_actual"]

# Prepare the input vector (no scaling)
input_row = [
    T_24,
    Df_1,
    Dr_24,
    P_24
]

# === ANFIS Model Definition ===
class BellMF(nn.Module):
    def __init__(self, in_features, mf_per_input):
        super().__init__()
        self.a = nn.Parameter(torch.ones(in_features, mf_per_input))
        self.b = nn.Parameter(torch.ones(in_features, mf_per_input))
        # Initialize centers from 0..1, but you can change this as needed
        self.c = nn.Parameter(torch.linspace(0, 1, mf_per_input).repeat(in_features, 1))

    def forward(self, x):
        # x shape: [batch_size, in_features]
        x_expanded = x.unsqueeze(2)  # -> [batch_size, in_features, 1]
        a = self.a.unsqueeze(0)      # -> [1, in_features, mf_per_input]
        b = self.b.unsqueeze(0)
        c = self.c.unsqueeze(0)
        # Bell membership function
        return 1.0 / (1.0 + torch.abs((x_expanded - c) / a)**(2*b))

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
        mf_values = self.mf_layer(x)              # [batch_size, in_features, mf_per_input]
        combined = mf_values.reshape(x.size(0), -1)  # Flatten membership outputs
        return self.rules(combined)

# === Prediction function ===
def predict_single_weather_point(input_values):
    # Convert to tensor of shape [1, in_features]
    input_tensor = torch.tensor(input_values, dtype=torch.float32).unsqueeze(0)

    final_prediction = 0.0
    for i in range(N_IMFS):
        # Load each IMF model
        model = SimpleANFIS(IN_FEATURES, MF_PER_INPUT)
        model.load_state_dict(torch.load(f"{MODEL_PREFIX}{i+1}{MODEL_EXT}"))
        model.eval()

        with torch.no_grad():
            final_prediction += model(input_tensor).item()

    return final_prediction

# === Run prediction ===
forecast = predict_single_weather_point(input_row)
actual   = df.loc[ROW_INDEX, "GR_solar_generation_actual"]

print(f"Past values used:")
print(f"  Temperature (t-24): {T_24}")
print(f"  Diffuse   (t-1):   {Df_1}")
print(f"  Direct    (t-24):  {Dr_24}")
print(f"  Solar Pow (t-24):  {P_24}")
print()
print(f"🔮 Forecasted solar generation: {forecast:.6f}")
print(f"🎯 Actual solar generation:     {actual:.6f}")
