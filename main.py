import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler

from utils.dataset import PhysionetDataset, AmbientaDataset
from utils.model import ConvNet
from utils.transforms import Blur, Close, Erode, ToTensor

################
#
# Hyper Parameters
#
################

num_epochs = 20
learning_rate = 0.001
batch_size = 100
train_size_percentage = 0.8

################
#
# Data Reading & Preprocessing
#
################

composed_transforms = torchvision.transforms.Compose(
    [Blur((5, 5)), Erode(), Close(), ToTensor()]
)

train_dataset = AmbientaDataset(composed_transforms, train=False)
test_dataset = AmbientaDataset(composed_transforms, train=True)

print(train_dataset.classes)
print(np.max(train_dataset.x))

# Over- & Undersampling
_, class_counts = np.unique(train_dataset.y, return_counts=True)
print(class_counts)
print(class_counts.sum())
weights = np.asarray([1.0 / class_counts[c] for c in train_dataset.y])
train_sampler = WeightedRandomSampler(
    weights=weights, num_samples=len(weights), replacement=True
)

# train_size = int(train_size_percentage * len(dataset))
# test_size = len(dataset) - train_size
# train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

sample, label = train_dataset[0]
print(sample.shape, label)

################
#
# Machine Learning Part
#
################

num_classes = len(train_dataset.classes)

# device config
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = ConvNet(num_classes)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

n_total_steps = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        labels = labels.long()
        # print(images.shape) # torch.Size([4, 1, 64, 32])
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        # print(outputs.shape, labels.shape) # torch.Size([4, 5]) torch.Size([4])
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % 10 == 0:
            print(
                f"Epoch {epoch+1} / {num_epochs}, step {i+1}/{n_total_steps}, loss = {loss.item():.4f}"
            )


with torch.no_grad():
    n_correct = 0
    n_samples = 0
    n_class_correct = [0 for i in range(num_classes)]
    n_class_samples = [0 for i in range(num_classes)]
    for images, labels in test_loader:
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

    acc = 100.0 * n_correct / n_samples
    print(n_class_correct)
    print(n_class_samples)

    print(f"Accuracy of the network: {acc:.4f}")
    for i in range(num_classes):
        acc = 100.0 * n_class_correct[i] / n_class_samples[i] if n_class_samples[i] != 0.0 else 0.0
        print(f"Accuracy of {test_dataset.classes[i]}: {acc:.4f}%")


################
#
# Data Visualization
# Can use both matplotlib (looks better) or PIL
#
################

# Transform Images back to arrays to visualize them with maptlotlib because it has better visualization
frame = np.asarray(sample[0])

# Visualize 2D Pressure Image as heatmap, cmap specifies the color scheme for the plot
plt.subplot(2, 3, 1)
plt.imshow(frame, origin="lower", cmap="gist_stern")
# plt.subplot(1, 2, 2)
# plt.imshow(frame_blur, origin="lower", cmap="gist_stern")
# plt.show()
