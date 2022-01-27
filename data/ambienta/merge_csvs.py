from importlib.metadata import files
import os
import glob
import pandas as pd
import pathlib
import numpy as np
import matplotlib.pyplot as plt
import progressbar

recordings_path = pathlib.Path(__file__).parent.joinpath("3").resolve()
print(recordings_path)
extension = "csv"
all_filenames = [i for i in glob.glob(f"{recordings_path}\*.{extension}")]
files_data = []

widgets = [
    "Reading Files: ",
    progressbar.Bar(left="[", right="]", marker="-"),
    " ",
    progressbar.Counter(format="%(value)02d/%(max_value)d"),
]

with progressbar.ProgressBar(max_value=len(all_filenames), widgets=widgets) as bar:
    for i, file in enumerate(all_filenames[:100]):
        filedata = np.loadtxt(file, delimiter=",", dtype=np.float32)
        files_data.append(filedata)
        bar.update(i + 1)

files_data = np.swapaxes(files_data, 1, 2)
files_data = np.flip(files_data, (1, 2))
files_data = np.reshape(files_data, (-1, files_data.shape[1] * files_data.shape[2]))
# print(np.max(files_data))
# files_data = files_data * (255/np.max(files_data))


# np.savetxt("combined_csv.csv", files_data, delimiter=",")

# files_data = [pd.DataFrame(d) for d in files_data]
# combined_csv = pd.concat(files_data)
# print(combined_csv)
# combined_csv.to_csv( "combined_csv.csv", index=False, encoding='utf-8-sig')
plt.subplot(1, 1, 1)
# plt.imshow(first_file, origin="lower", cmap="gist_stern")
# plt.show()
