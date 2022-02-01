import matplotlib.pyplot as plt
import numpy as np
import torch
import cv2
import torch.nn as nn
import torchvision
from torch.utils.data import (
    DataLoader,
    WeightedRandomSampler,
    ConcatDataset,
)

from utils.dataset import PhysionetDataset, AmbientaDataset, classes
from utils.model import ConvNet
from utils.transforms import (
    Blur,
    Close,
    Erode,
    ToTensor,
    Resize,
    Threshold,
    Normalize,
    EqualizeHist,
)
from sklearn.metrics import confusion_matrix
from utils.plots import plot_confusion_matrix, plot_class_weights

################
#
# Hyper Parameters
#
################

num_trainings = 10
num_epochs = 20
learning_rate = 0.005
batch_size = 100
num_classes = len(classes)

def main():

    ################
    #
    # Data Reading & Preprocessing
    #
    ################

    composed_transforms = torchvision.transforms.Compose(
        [
            Resize((26, 64), cv2.INTER_LINEAR),
            Normalize(),
            Resize((52, 128), cv2.INTER_LINEAR),
            ToTensor(),
        ]
    )

    train_dataset = ConcatDataset(
        [
            PhysionetDataset(composed_transforms, train=True),
            AmbientaDataset(composed_transforms, train=True),
        ]
    )

    test_dataset = ConcatDataset(
        [
            PhysionetDataset(composed_transforms, train=False),
            AmbientaDataset(composed_transforms, train=False),
        ]
    )

    print(f"Number of training samples: {len(train_dataset)}")
    print(f"Number of testing samples: {len(test_dataset)}")

    # Over- & Undersampling
    train_labels = np.concatenate(
        [train_dataset.datasets[0].y, train_dataset.datasets[1].y]
    )
    test_labels = np.concatenate([test_dataset.datasets[0].y, test_dataset.datasets[1].y])
    _, train_class_counts = np.unique(train_labels, return_counts=True)
    _, test_class_counts = np.unique(test_labels, return_counts=True)

    # plot_class_weights(
    #     [
    #         train_class_counts / train_class_counts.sum(),
    #         test_class_counts / test_class_counts.sum(),
    #     ],
    #     classes,
    #     [
    #         "Training Data Class Weights",
    #         "Test Data Class Weights",
    #     ],
    # )
    # plt.show()

    weights = np.asarray([1.0 / train_class_counts[c] for c in train_labels])
    train_sampler = WeightedRandomSampler(
        weights=weights, num_samples=len(weights), replacement=True
    )
    
    conf_mat_sum = np.zeros((11, 11))
    for i in range(num_trainings):
        conf_mat, acc = train_model(train_dataset, test_dataset, train_sampler)
        print(f"Accuracy of {i+1}. Network: {acc:.4f}")
        conf_mat_sum += conf_mat

    print(conf_mat_sum)
    with open('benchmarks/no_image_filters.npy', 'wb') as f:
        np.save(f, conf_mat_sum)

    # with open('test.npy', 'rb') as f:
    #     a = np.load(f)

    plot_confusion_matrix(conf_mat_sum, classes, normalize=True)
    plt.show()



def train_model(train_dataset, test_dataset, train_sampler):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    ################
    #
    # Machine Learning Part
    #
    ################

    # device config
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ConvNet(num_classes).to(device)

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

            if (i + 1) % 50 == 0:
                print(
                    f"Epoch {epoch+1} / {num_epochs}, step {i+1}/{n_total_steps}, loss = {loss.item():.4f}"
                )


    with torch.no_grad():
        n_correct = 0
        n_samples = 0
        predlist = []
        lbllist = []
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)

            _, predictions = torch.max(outputs, 1)

            lbllist.append(labels.cpu().numpy())
            predlist.append(predictions.cpu().numpy())
            n_samples += labels.size(0)
            n_correct += (predictions == labels).sum().item()

        acc = 100.0 * n_correct / n_samples

        return confusion_matrix(np.concatenate(lbllist), np.concatenate(predlist)), acc
        
        
if __name__ == "__main__":
    main()