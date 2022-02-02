from torch import conv2d
import torch.nn as nn
import torch.nn.functional as F


class ConvNet(nn.Module):
    def __init__(self, output_size):
        super(ConvNet, self).__init__()
        self.cnn1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
            nn.ReLU(inplace=True),
            # nn.Dropout2d(0.1),
            nn.MaxPool2d(3, stride=2),
            nn.Conv2d(64, 64, 3),
            nn.ReLU(inplace=True),
            # nn.Dropout2d(0.1),
            nn.Conv2d(64, 128, 3),
            nn.ReLU(inplace=True),
            # nn.Dropout2d(0.1),
            nn.Conv2d(128, 128, 3),
            nn.ReLU(inplace=True),
            # nn.Dropout2d(0.1),
        )

        self.cnn2_1 = nn.Sequential(
            nn.Conv2d(1, 32, 5),
            nn.ReLU(),
            nn.MaxPool2d(3),
        )

        self.cnn2_2 = nn.Sequential(
            nn.Conv2d(32, 32, 3),
            nn.ReLU(),
            nn.MaxPool2d(3),
        )
        
        self.cnn2_3 = nn.Sequential(
            nn.Conv2d(32, 64, 3),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3),
            nn.ReLU(),
            # nn.MaxPool2d(3),
        )
            
        self.cnn2_4 = nn.Sequential(
            nn.Dropout2d(0.1),
            nn.Conv2d(64, 1024, (12, 7)),
            nn.ReLU(),
        )
            
        self.cnn2_5 = nn.Sequential(
            nn.Conv1d(1024, 2048, 1),
            nn.ReLU(),
            nn.Conv1d(2048, output_size, 1),
        )

        self.fc = nn.Sequential(
            nn.Linear(128 * 25 * 6, output_size),
        )

        self.fc2 = nn.Sequential(
            nn.Linear(11 * 26 * 6, output_size),
        )

    def forward(self, x):
        x = self.cnn2_1(x)
        x = F.normalize(x)
        # x = self.cnn2_2(x)
        # x = F.normalize(x)
        x = self.cnn2_3(x)
        x = F.normalize(x)
        x = self.cnn2_4(x)
        x = F.normalize(x)
        # print(x.shape)
        x = x.view(-1, 1024, 26 * 6)
        x = self.cnn2_5(x)
        x = x.view(-1, 11 * 26 * 6)
        x = self.fc2(x)
        # print(x.shape)
        
        # x = self.cnn1(x)
        # x = x.view(-1, 128 * 25 * 6)
        # x = self.fc(x)
        return x
