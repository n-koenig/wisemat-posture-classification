import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageFilter
import cv2

################
#
# Data Reading
#
################

# Path to dataset
directory = "/home/nils/Documents/Studium/Veranstaltungen/WiSe_21_22/BP/Vortraege/a-pressure-map-dataset-for-in-bed-posture-classification-1.0.0/"

# Read first text file which contains 82 Pressure Images Frames, each is stored as one line of integer values
df = pd.read_csv(directory + "experiment-i/S1/1.txt", sep="\t", header=None)
# Drop last column which always contains NaN values because of bad formatting in text file
df = df.iloc[:, :2048]

# Alternative method for reading data
# usecols makes sure that last column is skipped, skiprows and max_rows is used to select which frame(s) are read
raw_frame = np.loadtxt(directory + "experiment-i/S1/1.txt", delimiter="\t", usecols=([_ for _ in range(2048)]), skiprows=80, max_rows=1)

# Transform Pressure Image from one line into 2D List
frame = [0 for _ in range(64)]
for i in range(64):
    frame[63-i] = raw_frame[i*32:(i+1)*32] # (df.iloc[2, i*32:(i+1)*32].values.tolist())

################
#
# Data Preprocessing
# Input is given as 2D array in "frame"
# 
################

# Transform 2D List into Image and apply Gaussian blur
# Numpy array needed for PIL package
frame = np.asarray(frame, dtype=np.uint8)
image = Image.fromarray(frame)
image_blur = image.filter(ImageFilter.GaussianBlur)
kernel = np.ones((5, 5), np.uint8)
image_bridge = cv2.morphologyEx(image_blur, cv2.MORPH_OPEN, kernel, )

# image.save('Positionserkennung/test.png')
# image.save('Positionserkennung/test_blur.png')

################
#
# Data Visualization
# Can use both matplotlib (looks better) or PIL
# 
################

# Transform Images back to arrays to visualize them with maptlotlib because it has better visualization
frame = np.asarray(image)
frame_blur = np.asarray(image_bridge)
# Alternative:
# iamae.show()
# image_blur.show()

# Visualize 2D Pressure Image as heatmap, cmap specifies the color scheme for the plot
plt.subplot(1, 2, 1)
plt.imshow(frame, origin="lower", cmap="gist_stern")
plt.subplot(1, 2, 2)
plt.imshow(frame_blur, origin="lower", cmap="gist_stern")
plt.show()
