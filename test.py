import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torchvision
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageFilter
import cv2
import torch
import torch.nn.functional as F
from functools import reduce
from sklearn.utils import shuffle

################
#
# Data Reading & Preprocessing
#
################

# Path to dataset
directory = "./wisemat-posture-classification/a-pressure-map-dataset-for-in-bed-posture-classification-1.0.0/"

labels_for_file = [0, 1, 2, 6, 6, 7, 7, 0, 0, 0, 0, 0, 3, 4, 5, 5, 5]
classes = ('Supine', 'Right', 'Left', 'Right Fetus', 'Left Fetus', 'Supine Bed Incline', 'Right Body Roll', 'Left Body Roll')
num_classes = len(classes)

class PressureDataset(Dataset):
    def __init__(self, transform=None):
        x_tensors = []
        y_tensors = []
        for subject in range(1, 14):
            for file in range(1, 18):
                # usecols makes sure that last column is skipped, skiprows is used to select which frame(s) are read
                raw_frames = np.loadtxt(f"{directory}experiment-i/S{subject}/{file}.txt", delimiter="\t", usecols=([_ for _ in range(2048)]), skiprows=2)
                x_tensors.append(np.flip(np.reshape(raw_frames, (raw_frames.shape[0], 64, 32)), 1))
                y_tensors.append(np.full([raw_frames.shape[0]], labels_for_file[file-1]))
                print(f"Subject {subject}, File {file} completed, {reduce(lambda count, l: count + len(l), y_tensors, 0)} samples in dataset")

        self.x = np.concatenate(x_tensors)
        self.y = np.concatenate(y_tensors)
        self.x, self.y = shuffle(self.x, self.y, random_state=234950)
        self.y = self.y.reshape(-1, 1)
        self.n_samples = self.x.shape[0]

        self.transform = transform

    def __getitem__(self, index):
        sample = self.x[index], self.y[index]
        if self.transform:
            sample = self.transform(sample)
        return sample

    def __len__(self):
        return self.n_samples
    

class ToTensor:
    def __call__(self, sample):
        inputs, targets = sample
        return torch.from_numpy(inputs), torch.from_numpy(targets)

class Blur:
    def __init__(self, ksize) -> None:
        self.ksize = ksize

    def __call__(self, sample):
        image, label = sample
        frame_blur = cv2.GaussianBlur(image, self.ksize, cv2.BORDER_DEFAULT)
        return frame_blur, label

class Threshold:
    def __call__(self, sample):
        image, label = sample
        th, frame_thresh = cv2.threshold(image, np.median(image), 255, cv2.THRESH_BINARY)
        return frame_thresh, label

class Erode:
    def __call__(self, sample):
        image, label = sample
        frame_eroded = cv2.erode(image, None, iterations=1)
        return frame_eroded, label

class Close:
    def __call__(self, sample):
        image, label = sample
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        frame_closed = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
        return frame_closed, label


composed_transforms = torchvision.transforms.Compose([ToTensor()])
batch_size = 100

train_dataset = PressureDataset(composed_transforms)
print(train_dataset.x.shape)
print(train_dataset.y.shape)

train_size = int(0.1 * len(train_dataset))
test_size = len(train_dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(train_dataset, [train_size, test_size])

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)

sample, label = train_dataset[0]
print(sample.shape, label)

################
#
# Machine Learning Part
# 
################

# device config
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# hyper parameters
num_epochs = 4
learning_rate = 0.001

# model definition
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16*13*5, 84)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, len(classes))

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16*13*5)
        x = F.relu(self.fc1(x))
        #x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

model = ConvNet()

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

n_total_steps = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        # print(images.shape)
        images = images.reshape((-1, 1, 64, 32))
        images = images.float()
        labels = labels.reshape((-1))
        # print(images.shape) # torch.Size([4, 1, 64, 32])
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        # print(outputs.shape, labels.shape) # torch.Size([4, 5]) torch.Size([4])
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 1 == 0:
            print(f"Epoch {epoch+1} / {num_epochs}, step {i+1}/{n_total_steps}, loss = {loss.item():.4f}")


with torch.no_grad():
    n_correct = 0
    n_samples = 0
    n_class_correct = [0 for i in range(num_classes)]
    n_class_samples = [0 for i in range(num_classes)]
    for images, labels in test_loader:
        images = images.reshape((-1, 1, 64, 32))
        images = images.float()
        labels = labels.reshape((-1))
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        # print(outputs)

        _, predictions = torch.max(outputs, 1)
        # print(_, predictions)
        n_samples += labels.size(0)
        n_correct += (predictions == labels).sum().item()

        # print(labels)
        for i in range(labels.shape[0]):
            label = labels[i]
            pred = predictions[i]
            if label == pred:
                n_class_correct[label] += 1
            n_class_samples[label] += 1

    acc = 100.0 * n_correct/n_samples
    print(n_class_correct)
    print(n_class_samples)
    print(sum(n_class_samples))

    print(f'Accuracy of the network: {acc}')
    for i in range(num_classes):
        acc = 100.0 * n_class_correct[i]/n_class_samples[i]
        print(f'Accuracy of {classes[i]}: {acc}%')


################
#
# Data Visualization
# Can use both matplotlib (looks better) or PIL
# 
################

# Transform Images back to arrays to visualize them with maptlotlib because it has better visualization
frame = np.asarray(sample)
# frame_blur = np.asarray(image_blur)
# Alternative:
# iamae.show()
# image_blur.show()

# Visualize 2D Pressure Image as heatmap, cmap specifies the color scheme for the plot
plt.subplot(1, 2, 1)
plt.imshow(frame, origin="lower", cmap="gist_stern")
# plt.subplot(1, 2, 2)
# plt.imshow(frame_blur, origin="lower", cmap="gist_stern")
# plt.show()
