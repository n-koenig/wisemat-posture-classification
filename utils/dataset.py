import numpy as np
from functools import reduce
from sklearn.utils import shuffle
from torch.utils.data import Dataset

class PressureDataset(Dataset):
    labels_for_file = [0, 1, 2, 6, 6, 7, 7, 0, 0, 0, 0, 0, 3, 4, 5, 5, 5]
    directory = "./a-pressure-map-dataset-for-in-bed-posture-classification-1.0.0/"
    classes = ('Supine', 'Right', 'Left', 'Right Fetus', 'Left Fetus', 'Supine Bed Incline', 'Right Body Roll', 'Left Body Roll')

    def __init__(self, transform=None):
        x_tensors = []
        y_tensors = []
        for subject in range(1, 2):
            for file in range(1, 18):
                # usecols makes sure that last column is skipped, skiprows is used to select which frame(s) are read
                raw_frames = np.loadtxt(f"{self.directory}experiment-i/S{subject}/{file}.txt", delimiter="\t", usecols=([_ for _ in range(2048)]), skiprows=2, dtype=np.float32)
                x_tensors.append(np.flip(np.reshape(raw_frames, (raw_frames.shape[0], 64, 32)), 1))
                y_tensors.append(np.full([raw_frames.shape[0]], self.labels_for_file[file-1]))
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
