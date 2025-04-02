import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from pathlib import Path

# === Load data ===
df = pd.read_csv(r"C:\Users\denys\Desktop\My-Projects\TUKE\Y2S2\fuzzy\data\complete_data.csv")
imfs = np.load(r"C:\Users\denys\Desktop\My-Projects\TUKE\Y2S2\fuzzy\data\imfs.npy")  # Do NOT slice here

# === Create lagged input features ===
features = pd.DataFrame({
    "T(t-24)": df["GR_temperature"].shift(24),
    "Df(t-1)": df["GR_radiation_diffuse_horizontal"].shift(1),
    "Dr(t-24)": df["GR_radiation_direct_horizontal"].shift(24),
    "P(t-24)": df["GR_solar_generation_actual"].shift(24),
})

# === Align inputs and targets ===
valid_idx = features.dropna().index
X = features.loc[valid_idx].reset_index(drop=True).values
Y = imfs[:, valid_idx].T  # shape [samples, 11]

# Convert to tensors
X_tensor = torch.tensor(X, dtype=torch.float32)

# === ANFIS Model ===
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

# === Training config ===
in_features = X.shape[1]
mf_per_input = 3
epochs = 300
lr = 0.01
n_imfs = Y.shape[1]
Path("models").mkdir(exist_ok=True)
criterion = nn.SmoothL1Loss()

# === Train all models ===
for i in range(n_imfs):
    print(f"\n🔁 Training ANFIS for IMF {i+1}...")
    Y_tensor = torch.tensor(Y[:, i].reshape(-1, 1), dtype=torch.float32)

    model = SimpleANFIS(in_features, mf_per_input)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        optimizer.zero_grad()
        output = model(X_tensor)
        loss = criterion(output, Y_tensor)
        loss.backward()
        optimizer.step()
        if epoch % 50 == 0:
            print(f"Epoch {epoch:3d} | Loss: {loss.item():.6f}")

    torch.save(model.state_dict(), f"models/anfis_imf{i+1}.pt")
    print(f"✅ Saved model to models/anfis_imf{i+1}.pt")

print("\n✅ All models trained and saved.")
