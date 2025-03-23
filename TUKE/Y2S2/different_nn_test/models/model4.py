from torch import nn

class Model4(nn.Module):
    def __init__(self, input_size, output_size):
        super(Model4, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, output_size)
        )

    def forward(self, x):
        return self.model(x)
