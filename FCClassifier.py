import torch.nn as nn

import torch.nn.functional as F


class FullyConnectedClassifier(nn.Module):
    def __init__(self):
        super(FullyConnectedClassifier, self).__init__()
        self.fc1 = nn.Linear(13, 25)
        self.fc2 = nn.Linear(25, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        h = self.fc2(x)
        x = F.relu(h)
        x = self.fc3(x)
        x = F.relu(x)
        x = self.fc4(x)
        return x
