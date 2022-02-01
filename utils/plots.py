import itertools
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

# from https://deeplizard.com/learn/video/0LhiS6yu2qQ
def plot_confusion_matrix(
    cm, classes, normalize=False, title="Confusion matrix", cmap=plt.cm.Oranges
):
    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print("Confusion matrix, without normalization")

    print(cm)
    plt.figure(figsize=(10, 10))
    ax = plt.gca()
    im = ax.imshow(cm, interpolation="nearest", cmap=cmap)
    plt.title(title)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = ".2f" if normalize else "d"
    thresh = cm.max() / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(
            j,
            i,
            format(cm[i, j], fmt),
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black",
        )

    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
    plt.tight_layout()


def plot_class_weights(weights_list, classes, legend_labels=[]):
    legend_labels += [None] * len(weights_list)
    bar_width = 0.8 / len(weights_list)
    x = np.arange(len(classes))
    for i, (weights, label) in enumerate(zip(weights_list, legend_labels)):
        bar = plt.bar(x + i * bar_width, weights, bar_width)
        if label:
            bar.set_label(label)

    plt.title("Relative Classes distribution")

    print(x + bar_width / 2)
    plt.xticks(x + bar_width / 2, classes, rotation=45)
    plt.ylabel("Class Weight")
    plt.tight_layout()

    if legend_labels[0]:
        plt.legend()
