from matplotlib import transforms
import matplotlib.pyplot as plt
import numpy as np
import torch
import cv2
import random
import torchvision

from utils.dataset import PhysionetDataset, AmbientaDataset
from utils.transforms import (
    Blur,
    Close,
    Erode,
    ToTensor,
    Resize,
    Threshold,
    EqualizeHist,
    Normalize,
)

composed_transforms = torchvision.transforms.Compose(
    [
        # Resize((26, 64), cv2.INTER_LINEAR),
        # Blur((5, 5)),
        # Threshold(),
        # Erode(),
        # Close(),
        # ToTensor(),
    ]
)

train_dataset1 = PhysionetDataset(composed_transforms, train=False)
train_dataset2 = AmbientaDataset(composed_transforms, train=False)
test_dataset1 = PhysionetDataset(composed_transforms, train=True)
test_dataset2 = AmbientaDataset(composed_transforms, train=True)
print(len(train_dataset1))
print(len(train_dataset2))
print(len(test_dataset1))
print(len(test_dataset2))
trans = [
    Resize((26, 64), cv2.INTER_LINEAR),
    Normalize(),
    EqualizeHist(),
    Blur((5, 5)),
    Erode(),
    Erode(),
    Erode(),
]

trans_labels = [
    "Original",
    "Resize",
    "Normalize",
    "Equalize Histogram",
    "Gaussian Blur",
    "Erosion",
    "Erosion",
    "Erosion",
]

transform1 = torchvision.transforms.Compose(trans[:1])
transform2 = torchvision.transforms.Compose(trans[:2])
transform3 = torchvision.transforms.Compose(trans[:3])
transform4 = torchvision.transforms.Compose(trans[:4])
transform5 = torchvision.transforms.Compose(trans[:5])
transform6 = torchvision.transforms.Compose(trans[:6])
transform7 = torchvision.transforms.Compose(trans[:7])

num_plots_per_row = 4
images_per_dataset = 1

samples = [train_dataset1[128], train_dataset2[128]]
fig, axs = plt.subplots(2, 2)
for sample, row, row_nr in zip(samples, axs, range(len(axs))):
    for i, ax in enumerate(row):
        transform = torchvision.transforms.Compose(trans[:0 if i == 0 else 5])
        ax.imshow(transform(sample)[0][0], origin="lower", cmap="gist_stern")
        if row_nr == 0:
            ax.set_title(trans_labels[0 if i == 0 else 5])
        if i == 0 and row_nr == 0:
            ax.set_ylabel("Physionet")
        elif i == 0 and row_nr == 1:
            ax.set_ylabel("Ambienta")
        else:
            pass

plt.show()

for i in range(0, num_plots_per_row * images_per_dataset, num_plots_per_row):
    x = random.randint(0, len(train_dataset1) - 1)
    sample = train_dataset1[128]

    one = sample[0][0]
    two = transform1(sample)[0][0]
    three = transform2(sample)[0][0]
    four = transform3(sample)[0][0]

    plt.suptitle("Original")
    plt.subplot(images_per_dataset * 2, num_plots_per_row, i + 1)
    plt.imshow(one, origin="lower", cmap="gist_stern")

    plt.subplot(images_per_dataset * 2, num_plots_per_row, i + 2)
    plt.hist(one.ravel(), bins=50, color="#777777")
    # plt.imshow(two, origin="lower", cmap="gist_stern")

    plt.suptitle("Equalized Histogram")
    plt.subplot(images_per_dataset * 2, num_plots_per_row, i + 3)
    plt.imshow(four, origin="lower", cmap="gist_stern")

    plt.subplot(images_per_dataset * 2, num_plots_per_row, i + 4)
    # plt.imshow(four, origin="lower", cmap="gist_stern")
    plt.hist(four.ravel(), bins=50, color="#777777")
    continue
    five = transform4(sample)[0][0]
    plt.subplot(images_per_dataset * 2, num_plots_per_row, i + 5)
    plt.imshow(five, origin="lower", cmap="gist_stern")

    six = transform5(sample)[0][0]
    plt.subplot(images_per_dataset * 2, num_plots_per_row, i + 6)
    plt.imshow(six, origin="lower", cmap="gist_stern")

    seven = transform5(sample)[0][0]
    plt.subplot(images_per_dataset * 2, num_plots_per_row, i + 7)
    plt.imshow(seven, origin="lower", cmap="gist_stern")

    eight = transform5(sample)[0][0]
    plt.subplot(images_per_dataset * 2, num_plots_per_row, i + 8)
    plt.imshow(eight, origin="lower", cmap="gist_stern")

for i in range(
    num_plots_per_row * images_per_dataset,
    num_plots_per_row * images_per_dataset * 2,
    num_plots_per_row,
):
    x = random.randint(0, len(train_dataset2) - 1)
    sample = train_dataset2[128]

    one = sample[0][0]
    two = transform1(sample)[0][0]
    three = transform2(sample)[0][0]
    four = transform3(sample)[0][0]
    plt.subplot(images_per_dataset * 2, num_plots_per_row, i + 1)
    plt.imshow(one, origin="lower", cmap="gist_stern")

    plt.subplot(images_per_dataset * 2, num_plots_per_row, i + 2)
    plt.hist(one.ravel(), bins=50, color="#777777")
    # plt.imshow(two, origin="lower", cmap="gist_stern")

    plt.subplot(images_per_dataset * 2, num_plots_per_row, i + 3)
    plt.imshow(four, origin="lower", cmap="gist_stern")

    plt.subplot(images_per_dataset * 2, num_plots_per_row, i + 4)
    # plt.imshow(four, origin="lower", cmap="gist_stern")
    plt.hist(four.ravel(), bins=50, color="#777777")
    continue
    five = transform4(sample)[0][0]
    plt.subplot(images_per_dataset * 2, num_plots_per_row, i + 5)
    plt.imshow(five, origin="lower", cmap="gist_stern")

    six = transform5(sample)[0][0]
    plt.subplot(images_per_dataset * 2, num_plots_per_row, i + 6)
    plt.imshow(six, origin="lower", cmap="gist_stern")

    seven = transform5(sample)[0][0]
    plt.subplot(images_per_dataset * 2, num_plots_per_row, i + 7)
    plt.imshow(seven, origin="lower", cmap="gist_stern")

    eight = transform5(sample)[0][0]
    plt.subplot(images_per_dataset * 2, num_plots_per_row, i + 8)
    plt.imshow(eight, origin="lower", cmap="gist_stern")
