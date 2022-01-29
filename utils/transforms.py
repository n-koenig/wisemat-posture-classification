import torch
import cv2
import numpy as np
import torchvision

class ToTensor:
    def __call__(self, sample):
        image, label = sample
        return torch.from_numpy(image), label

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
        th, frame_thresh = cv2.threshold(image, np.median(image), 1, cv2.THRESH_BINARY)
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

class Resize:
    def __init__(self, size, interpolation) -> None:
        self.size = size
        self.interpolation = interpolation

    def __call__(self, sample):
        image, label = sample
        frame_resized = cv2.resize(image[0], self.size, interpolation=self.interpolation)
        return np.array([frame_resized]), label

class Normalize:
    def __call__(self, sample):
        x = sample / np.max(sample)
        mean = np.mean(x, (1, 2))
        std = np.std(x, (1, 2))
        # print(mean, std)
        return (x - x.mean(axis=(0,1,2), keepdims=True)) / x.std(axis=(0,1,2), keepdims=True)
        return np.asarray(torchvision.transforms.Normalize(mean, std)(torch.from_numpy(image)))
