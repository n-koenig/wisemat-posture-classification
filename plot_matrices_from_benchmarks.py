import numpy as np
from utils.plots import (
    plot_confusion_matrix,
    plot_comparing_confusion_matrix,
)
from utils.calculations import f1_scores_from_conf_mat
import matplotlib.pyplot as plt
from utils.dataset import classes


# define and load current benchmark values
current = 'normalize_equalize'
# sum_conf = np.zeros((11, 11))
# for i in range(10):
#     with open(f'benchmarks/{current}_{i+1}.npy', 'rb') as f:
#         conf = np.load(f)
#         print(conf)
#         sum_conf += conf
# conf_mat = sum_conf / 10


conf_mat = []
with open(f'benchmarks/{current}.npy', 'rb') as f:
    conf_mat = np.load(f)

with open(f'benchmarks/baseline.npy', 'rb') as f:
    base_conf_mat = np.load(f)

# plot and save confusion matrix with and without comparison
# plot_confusion_matrix(conf_mat, classes, normalize=True)
# plt.savefig(f'images/confusion_matrices/{current}_transparent.png', transparent=True)
# plt.show()
# plot_comparing_confusion_matrix(base_conf_mat, conf_mat, classes, normalize=True)
# plt.savefig(f'images/confusion_matrices/{current}_compare_transparent.png', transparent=True)
# plt.show()

# compute and save f1 scores
f1_scores = f1_scores_from_conf_mat(conf_mat)
f1_scores.append(np.mean(f1_scores))
np.savetxt(f'benchmarks/{current}_f1.txt', f1_scores, "%.2f", delimiter='\n')
exit()
# with relation to baseline
base_f1_scores = f1_scores_from_conf_mat(base_conf_mat)
base_f1_scores.append(np.mean(base_f1_scores))

f1_scores_fmt = []
for i in range(len(f1_scores)):
    if (f1_scores[i] - base_f1_scores[i]) > 0:
        f1_scores_fmt.append(f'{"%.2f" % f1_scores[i]} (+{"%.2f" % (f1_scores[i] - base_f1_scores[i])})')
    elif (f1_scores[i] - base_f1_scores[i]) < 0:
        f1_scores_fmt.append(f'{"%.2f" % f1_scores[i]} ({"%.2f" % (f1_scores[i] - base_f1_scores[i])})')
with open(f'benchmarks/{current}_f1.txt', 'w') as f:
    f.writelines("%s\n" % item for item in f1_scores_fmt)
