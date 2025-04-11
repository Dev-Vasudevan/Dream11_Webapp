import torch
from torch import nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin1 = nn.Linear(28, 128)
        self.lin3 = nn.Linear(128, 64)
        self.lin4 = nn.Linear(64, 32)
        self.lin5 = nn.Linear(32, 1)
        self.relu = nn.LeakyReLU()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(p=0.3)  # Adjust p based on validation performance

    def forward(self, x):
        x = self.lin1(x)
        x = self.sigmoid(x)
        x = self.dropout(x)  # After first activation

        x = self.lin3(x)
        x = self.relu(x)
        x = self.dropout(x)  # After second activation

        x = self.lin4(x)
        x = self.relu(x)
        x = self.dropout(x)  # After third activation

        x = self.lin5(x)
        return x
