import torch
import cv2
import numpy as np

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
