import itertools
import numpy as np
import matplotlib as mpl
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

    plt.figure(figsize=(10, 10))
    ax = plt.gca()
    im = ax.imshow(cm, interpolation="nearest", cmap=cmap)
    # plt.title(title)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
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
    im.axes.tick_params(color="#CACACA", labelcolor="#CACACA")
    ax.xaxis.label.set_color("#CACACA")
    ax.yaxis.label.set_color("#CACACA")
    plt.xlabel("Predicted label")

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.25)
    norm = mpl.colors.Normalize(vmin=-1, vmax=1)
    cb = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax)
    cb.ax.yaxis.set_tick_params(color="#CACACA")
    cb.outline.set_edgecolor("#CACACA")
    plt.setp(plt.getp(cb.ax.axes, "yticklabels"), color="#CACACA")

    for spine in im.axes.spines.values():
        spine.set_edgecolor("#CACACA")

    plt.tight_layout()


def plot_comparing_confusion_matrix(
    base_cm, compare_cm, classes, normalize=False, title="Confusion matrix"
):
    if normalize:
        base_cm = base_cm.astype("float") / base_cm.sum(axis=1)[:, np.newaxis]
        compare_cm = compare_cm.astype("float") / compare_cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print("Confusion matrix, without normalization")

    cm = compare_cm - base_cm
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if i != j:
            cm[i, j] *= -1

    plt.figure(figsize=(10, 10))
    cmap = mpl.colors.LinearSegmentedColormap.from_list(
        "rg", ["r", "r", "w", "g", "g"], N=256
    )
    ax = plt.gca()
    im = ax.imshow(cm, interpolation="nearest", cmap=cmap, vmin=-1, vmax=1)
    # plt.title(title)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)

    fmt = ".2f" if normalize else "d"
    thresh = cm.max() / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        n = cm[i, j] if i == j else cm[i, j] * -1
        number = float(int(n)) if int(n * 100) == 0 else n
        plt.text(
            j,
            i,
            f"{'+' if number > 0 else ''}{format(number, fmt)}",
            horizontalalignment="center",
            color="black",
            weight="bold" if i == j else "normal",
        )

    plt.ylabel("True label")
    im.axes.tick_params(color="#CACACA", labelcolor="#CACACA")
    ax.xaxis.label.set_color("#CACACA")
    ax.yaxis.label.set_color("#CACACA")
    plt.xlabel("Predicted label")

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.25)
    norm = mpl.colors.Normalize(vmin=-1, vmax=1)
    cb = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax)
    cb.ax.yaxis.set_tick_params(color="#CACACA")
    cb.outline.set_edgecolor("#CACACA")
    plt.setp(plt.getp(cb.ax.axes, "yticklabels"), color="#CACACA")

    for spine in im.axes.spines.values():
        spine.set_edgecolor("#CACACA")

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
