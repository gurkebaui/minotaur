import torch
import torch.nn as nn

class ImageClassifier(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc_1 = nn.Linear(3072, 512)
        self.relu = nn.ReLU()
        self.fc_2 = nn.Linear(512, 10)

    def forward(self, x):
        x = self.fc_1(x)
        x = self.relu(x)
        x = self.fc_2(x)
        return x


