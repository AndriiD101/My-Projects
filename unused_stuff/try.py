import torch

# Check if CUDA (GPU support) is available
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("✅ GPU is available. Using:", torch.cuda.get_device_name(0))
else:
    device = torch.device("cpu")
    print("❌ GPU not available. Using CPU.")

# Optional: Show current device being used
print("Current device:", device)

import torch
print(torch.version.cuda)     # Should show something like '11.8'
print(torch.backends.cudnn.version())  # Should return a version number, not None
