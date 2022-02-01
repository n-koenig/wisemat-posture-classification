import torch.nn as nn
import torch.nn.functional as F


class ConvNet(nn.Module):
    def __init__(self, output_size):
        super(ConvNet, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.MaxPool2d(3, stride=2),
            nn.Conv2d(64, 64, 3),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(64, 128, 3),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(128, 128, 3),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
        )

        self.fc = nn.Sequential(
            nn.Linear(128 * 25 * 6, output_size),
        )

    def forward(self, x):
        x = self.cnn(x)
        x = x.view(-1, 128 * 25 * 6)
        x = self.fc(x)
        return x
