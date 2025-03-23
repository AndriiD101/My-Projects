import torch.nn as nn

class Model1(nn.Module):
    def __init__(self, input_size, output_size):
        super(Model1, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 16),
            nn.ReLU(),
            nn.Linear(16, output_size)
        )

    def forward(self, x):
        return self.model(x)

