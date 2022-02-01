from matplotlib import transforms
import matplotlib.pyplot as plt
import numpy as np
import torch
import cv2
import random
import torchvision

from utils.dataset import PhysionetDataset, AmbientaDataset, classes
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

train_dataset1 = PhysionetDataset(composed_transforms, train=True)
train_dataset2 = AmbientaDataset(composed_transforms, train=True)
test_dataset1 = PhysionetDataset(composed_transforms, train=False)
test_dataset2 = AmbientaDataset(composed_transforms, train=False)
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
    Resize((52, 128), cv2.INTER_LINEAR),
]

trans_labels = [
    "Original",
    "Resized",
    "Normalized",
    "Equalized Histogram",
    "Gaussian Blurred",
    "Eroded",
    "Eroded",
    "Eroded",
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

# samples = [train_dataset1[128], train_dataset2[128]]

# fig = plt.figure(figsize=(4, 2))
# subfigs = fig.subfigures(1, 2, wspace=0)

# subfigs[0].suptitle('Original', fontsize='x-large')
# axsLeft = subfigs[0].subplots(2, 2)
# # subfigs[0].set_facecolor('0.75')
# # subfigs[0].colorbar(pc, shrink=0.6, ax=axsLeft, location='bottom')
# for nn, row in enumerate(axsLeft):
#     sample = samples[0 if nn == 0 else 1]
#     if nn == 0:
#         row[0].set_ylabel('Physionet')
#     else:
#         row[0].set_ylabel('Ambienta')
#     transform = torchvision.transforms.Compose([])
#     row[0].imshow(transform(sample)[0][0], origin="lower", cmap="gist_stern")
#     row[0].set_xticks([], [])
#     row[0].set_yticks([], [])
#     if nn == 0:
#         row[0].set_title("Image")
#         row[1].set_title("Histogram")
#     row[1].hist(transform(sample)[0][0].ravel(), bins=50, color="#777777")

# subfigs[1].suptitle('Normalized Values', fontsize='x-large')
# axsRight = subfigs[1].subplots(2, 2)
# for nn, row in enumerate(axsRight):
#     transform = torchvision.transforms.Compose(trans[:2])
#     sample = samples[0 if nn == 0 else 1]
#     row[0].imshow(transform(sample)[0][0], origin="lower", cmap="gist_stern")
#     row[0].set_xticks([], [])
#     row[0].set_yticks([], [])
#     if nn == 0:
#         row[0].set_title("Image")
#         row[1].set_title("Histogram")
#     row[1].hist(transform(sample)[0][0].ravel(), bins=50, color="#777777")

# subfigs[1].set_facecolor('0.85') # Changes the background of the subfigure
# subfigs[1].colorbar(pc, shrink=0.6, ax=axsRight)

# fig.suptitle('Figure suptitle', fontsize='xx-large')


prone1 = []
for image, label in train_dataset2:
    if label == classes.index("Prone"):
        prone1.append(
            (
                image,
                label,
            )
        )
    if len(prone1) == 8:
        break

rlateral1 = []
for image, label in train_dataset1:
    if label == classes.index("Lateral_Left"):
        rlateral1.append(
            (
                image,
                label,
            )
        )
    if len(rlateral1) == 8:
        break

rlateral2 = []
for image, label in train_dataset2:
    if label == classes.index("Lateral_Left"):
        rlateral2.append(
            (
                image,
                label,
            )
        )
    if len(rlateral2) == 8:
        break

samples = [prone1, rlateral1, rlateral2]

fig, axs = plt.subplots(3, 8)
for row_samples, row, row_nr in zip(samples, axs, range(len(axs))):
    for i, ax in enumerate(row):
        sample = row_samples[i]
        transform = torchvision.transforms.Compose(trans)
        ax.imshow(transform(sample)[0][0], origin="lower", cmap="gist_stern")
        # if (i == 0 and row_nr == 0):
        #     ax.set_xticks(range(0, 31, 10), range(0, 31, 10))
        # else:
        #     ax.set_xticks(range(0, 26, 10), range(0, 26, 10))
        ax.set_xticks([], [])
        ax.set_yticks([], [])
        if row_nr == 0:
            ax.set_title(trans_labels[0 if i == 0 else 1])

axs[0][0].set_ylabel("Ambienta")
axs[1][0].set_ylabel("Physionet")
axs[2][0].set_ylabel("Ambienta")

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
