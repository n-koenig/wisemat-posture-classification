import random
import numpy as np
import pandas as pd
import cv2
from functools import reduce
from sklearn.utils import shuffle
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import progressbar
from utils.transforms import Resize

classes = (
        "Supine",
        "Lateral_Right",
        "Lateral_Left",
        "KneeChest_Right",
        "KneeChest_Left",
        "Supine Bed Incline",
        "Right Body Roll",
        "Left Body Roll",
        "SittingOnEdge",
        "SittingOnBed",
        "Prone",
    )


class PhysionetDataset(Dataset):
    labels_for_file = [0, 1, 2, 6, 6, 7, 7, 0, 0, 0, 0, 0, 3, 4, 5, 5, 5]
    directory = "./data/physionet/"
    classes2 = (
        "Supine",
        "Right",
        "Left",
        "Right Fetus",
        "Left Fetus",
        "Supine Bed Incline",
        "Right Body Roll",
        "Left Body Roll",
    )

    def __init__(self, transform=None, train=False):
        subjects = range(1, 9) if train else range(9, 14)
        records_per_subject = range(1, 18)
        self.x, self.y = self.read_files(subjects, records_per_subject)
        # filter = Resize((26, 64), cv2.INTER_LINEAR)
        # for i in range(0, 6, 2):
        #     x = random.randint(0, self.x.shape[0] - 1)
        #     plt.subplot(3, 2, i + 1)
        #     plt.imshow(self.x[x][0], origin="lower", cmap="gist_stern")
        #     plt.subplot(3, 2, i + 2)
        #     plt.imshow(filter(self.x[x][0]), origin="lower", cmap="gist_stern")
        # plt.show()
        self.x, self.y = shuffle(self.x, self.y, random_state=234950)

        self.n_samples = self.x.shape[0]

        self.transform = transform

    def __getitem__(self, index):
        sample = self.x[index], self.y[index]
        if self.transform:
            sample = self.transform(sample)
        return sample

    def __len__(self):
        return self.n_samples

    def read_files(self, subjects, records_per_subject):
        x_tensors = []
        y_tensors = []
        widgets = [
            "Reading Files: ",
            progressbar.Bar(left="[", right="]", marker="-"),
            " ",
            progressbar.Counter(format="%(value)02d/%(max_value)d"),
            ", ",
            progressbar.Variable(
                "samples", format="Total Samples: {formatted_value}", width=4
            ),
        ]

        with progressbar.ProgressBar(
            max_value=len(subjects) * len(records_per_subject), widgets=widgets
        ) as bar:
            for subject in subjects:
                for file in records_per_subject:
                    # usecols makes sure that last column is skipped, skiprows is used to select which frame(s) are read
                    raw_frames = np.loadtxt(
                        f"{self.directory}experiment-i/S{subject}/{file}.txt",
                        delimiter="\t",
                        usecols=([_ for _ in range(2048)]),
                        skiprows=2,
                        dtype=np.float32,
                    )
                    # print(raw_frames.shape)
                    raw_frames = np.reshape(raw_frames, (-1, 1, 64, 32))
                    raw_frames = np.flip(raw_frames, (2, 3))
                    x_tensors.append(raw_frames)
                    y_tensors.append(
                        np.full([raw_frames.shape[0]], self.labels_for_file[file - 1])
                    )
                    # print(
                    #     f"Subject {subject}, File {file} completed, {reduce(lambda count, l: count + len(l), y_tensors, 0)} samples in dataset"
                    # )
                    # for x in range(9):
                    #     i = random.randint(0, raw_frames.shape[0] - 1)
                    #     plt.subplot(3, 3, x + 1)
                    #     plt.imshow(raw_frames[i][0], origin="lower", cmap="gist_stern")
                    bar.update(
                        ((subject - subjects[0]) * len(records_per_subject)) + file,
                        samples=reduce(lambda count, l: count + len(l), y_tensors, 0),
                    )

        return np.concatenate(x_tensors), np.concatenate(y_tensors)


class AmbientaDataset(Dataset):
    directory = "./data/ambienta/"
    classes2 = [
        "Supine",
        "SittingOnEdge",
        "SittingOnBed",
        "Lateral_Right",
        "Prone",
        "Lateral_Left",
        "KneeChest_Left",
    ]

    def __init__(self, transform=None, train=False):
        x_arrays = []
        y_arrays = []
        subjects = range(3, 5) if train else range(5, 6)
        for subject in subjects:
            # usecols makes sure that last column is skipped, skiprows is used to select which frame(s) are read
            raw_frames = np.loadtxt(
                f"{self.directory}{subject}.gz", delimiter=",", dtype=np.float32
            )
            raw_frames = np.reshape(raw_frames, (-1, 1, 64, 26))

            raw_labels = pd.read_csv(f"{self.directory}{subject}_labels.csv")

            labels = []
            frames_to_remove = []
            for frame_nr, _, label in raw_labels.itertuples():
                if label in classes:
                    labels.append(classes.index(label))
                else:
                    frames_to_remove.append(frame_nr)
                    labels.append(-1)

            raw_frames = np.delete(raw_frames, frames_to_remove, 0)
            labels = np.delete(labels, frames_to_remove, 0)

            x_arrays.append(raw_frames)
            y_arrays.append(labels)

        self.x = np.concatenate(x_arrays)
        self.y = np.concatenate(y_arrays)
        self.x, self.y = shuffle(self.x, self.y, random_state=234950)
        self.n_samples = self.x.shape[0]

        self.transform = transform

    def __getitem__(self, index):
        sample = self.x[index], self.y[index]
        if self.transform:
            sample = self.transform(sample)
        return sample

    def __len__(self):
        return self.n_samples
