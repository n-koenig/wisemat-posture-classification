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
        frame_blur = cv2.GaussianBlur(image[0], self.ksize, cv2.BORDER_DEFAULT)
        return np.array([frame_blur]), label


class Threshold:
    def __call__(self, sample):
        image, label = sample
        th, frame_thresh = cv2.threshold(image, np.median(image), 1, cv2.THRESH_TOZERO)
        return frame_thresh, label


class Erode:
    def __call__(self, sample):
        image, label = sample
        frame_eroded = cv2.erode(image[0], None, iterations=1)
        return np.array([frame_eroded]), label


class Close:
    def __call__(self, sample):
        image, label = sample
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        frame_closed = cv2.morphologyEx(image[0], cv2.MORPH_CLOSE, kernel)
        return np.array([frame_closed]), label


class Resize:
    def __init__(self, size, interpolation) -> None:
        self.size = size
        self.interpolation = interpolation

    def __call__(self, sample):
        image, label = sample
        frame_resized = cv2.resize(
            image[0], self.size, interpolation=self.interpolation
        )
        return np.array([frame_resized]), label


class Normalize:
    def __call__(self, sample):
        image, label = sample
        return image / np.max(image), label
        mean = np.mean(x, (1, 2))
        std = np.std(x, (1, 2))
        # print(mean, std)
        return (x - x.mean(axis=(0, 1, 2), keepdims=True)) / x.std(
            axis=(0, 1, 2), keepdims=True
        )
        return np.asarray(
            torchvision.transforms.Normalize(mean, std)(torch.from_numpy(image))
        )


class EqualizeHist:
    def __call__(self, sample):
        image, label = sample
        image = image * 255.0
        image = image.astype(np.uint8)
        image = cv2.equalizeHist(image[0])
        image = image.astype(np.float32) / 255.0
        return np.array([image]), label
