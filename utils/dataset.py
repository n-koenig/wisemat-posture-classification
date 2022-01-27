import random
import numpy as np
import pandas as pd
from functools import reduce
from sklearn.utils import shuffle
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import progressbar


class PhysionetDataset(Dataset):
    labels_for_file = [0, 1, 2, 6, 6, 7, 7, 0, 0, 0, 0, 0, 3, 4, 5, 5, 5]
    directory = "./data/physionet/"
    classes = (
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
        subjects = range(9, 14) if train else range(1, 9)
        records_per_subject = range(1, 18)
        self.x, self.y = self.read_files(subjects, records_per_subject)
        self.x, self.y = shuffle(self.x, self.y, random_state=234950)
        # self.x = self.x * (255/4095.0)

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
            progressbar.Variable("samples", format='Total Samples: {formatted_value}', width=4),
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
                    raw_frames = np.flip(raw_frames, 2)
                    x_tensors.append(raw_frames)
                    y_tensors.append(
                        np.full([raw_frames.shape[0]], self.labels_for_file[file - 1])
                    )
                    # print(
                    #     f"Subject {subject}, File {file} completed, {reduce(lambda count, l: count + len(l), y_tensors, 0)} samples in dataset"
                    # )
                    for x in range(9):
                        i = random.randint(0, raw_frames.shape[0] - 1)
                        plt.subplot(3, 3, x + 1)
                        plt.imshow(raw_frames[i][0], origin="lower", cmap="gist_stern")
                    bar.update(
                        ((subject - subjects[0]) * len(records_per_subject)) + file,
                        samples=reduce(lambda count, l: count + len(l), y_tensors, 0),
                    )

        return np.concatenate(x_tensors), np.concatenate(y_tensors)


class AmbientaDataset(Dataset):
    directory = "./data/ambienta/"
    ignored_labels = [
        "NoPerson",
        "SitOnEdge",
        # "SittingOnEdge",
        "StandingFromBed",
        "LyingOnBed",
        # "SittingOnBed",
        "NotDefined",
        "Rotation_Supine_RLateral",
        "Rotation_Right_Prone",
        "Rotation_Prone_Right",
        "Rotation_Right_Supine",
        "Rotation_Supine_Left",
        "Changing",
        "Rotation_Supine_LLateral",
    ]

    def __init__(self, transform=None):
        x_arrays = []
        y_arrays = []
        for file in range(3, 4):
            # usecols makes sure that last column is skipped, skiprows is used to select which frame(s) are read
            raw_frames = np.loadtxt(
                f"{self.directory}{file}.gz", delimiter=",", dtype=np.float32
            )
            raw_frames = np.reshape(raw_frames, (-1, 1, 64, 26))
            raw_frames = np.flip(raw_frames, 2)

            raw_labels = pd.read_csv(f"{self.directory}{file}_labels.csv")
            self.classes = []

            labels = []
            frames_to_remove = []
            for frame_nr, _, label in raw_labels.itertuples():
                if label not in self.classes:
                    if label not in self.ignored_labels:
                        self.classes.append(label)
                        labels.append(len(self.classes) - 1)
                    else:
                        frames_to_remove.append(frame_nr)
                        labels.append(-1)
                else:
                    labels.append(self.classes.index(label))

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
