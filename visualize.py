import numpy as np
import matplotlib.pyplot as plt

titles = np.loadtxt("watt3.txt", dtype=str, max_rows=1)
titles = titles.tolist().split(',')
print(titles)


print(type(titles))
arr = np.genfromtxt("watt4.txt", delimiter=',', filling_values=0)

print(arr)
# print(np.where(arr==0)[0][0])
# accuracy = arr[np.where(arr==0)[0][0]][0]
# print(accuracy)
mid = np.where(arr==0)[0][7]
print(mid)
arr1 = arr[:mid, :]
arr2 = arr[mid:, :]
print(arr1)
print(arr2)

arr1 = np.delete(arr1, np.where(arr1==0)[0], 0)
arr2 = np.delete(arr2, np.where(arr2==0)[0], 0)

arr = np.delete(arr, np.where(arr==0)[0], 0)


# print(arr[:, 0])

# print(arr)


# plt.figure(1)
# plt.plot(arr)
# # plt.figure(2)
# fig, axs = plt.subplots(2, 2, sharex='all', sharey='all')
# axs[0, 0].plot(arr[:, 0])
# axs[0, 0].set_title(titles[0])
# axs[0, 1].plot(arr[:, 1])
# axs[0, 1].set_title(titles[1])
# axs[1, 0].plot(arr[:, 2])
# axs[1, 0].set_title(titles[2])
# axs[1, 1].plot(arr[:, 3])
# axs[1, 1].set_title(titles[3])



plt.figure(1)
plt.plot(arr)
# plt.plot(arr1, label="Run 1")
# plt.plot(arr2, label="Run 2")
# plt.legend()

# plt.figure(2)
fig, axs = plt.subplots(2, 2, sharex='all', sharey='all')
axs[0, 0].plot(arr1[:, 0], label="Run 1")
axs[0, 0].plot(arr2[:, 0], label="Run 1")
axs[0, 0].set_title(titles[0])
# plt.legend()
axs[0, 1].plot(arr1[:, 1], label="Run 1")
axs[0, 1].plot(arr2[:, 1], label="Run 2")
axs[0, 1].set_title(titles[1])
# plt.legend()
axs[1, 0].plot(arr1[:, 2], label="Run 1")
axs[1, 0].plot(arr2[:, 2], label="Run 2")
axs[1, 0].set_title(titles[2])
# plt.legend()
axs[1, 1].plot(arr1[:, 3], label="Run 1")
axs[1, 1].plot(arr2[:, 3], label="Run 2")
axs[1, 1].set_title(titles[3])
plt.legend()

plt.show()
