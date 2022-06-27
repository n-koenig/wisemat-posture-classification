import numpy as np
import matplotlib.pyplot as plt

arr = np.loadtxt("watts.txt", skiprows=1, delimiter=',')

print(arr)

plt.plot(arr)
plt.show()