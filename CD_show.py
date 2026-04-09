import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

flag = 'CD21'
root = '/data/xtw/Project/pythonProject/00one_clik/1406542/p3channel/' + flag

paths = os.listdir(root)
paths.sort()
print(paths)
for i in range(len(paths)):
    pathi = paths[i]
    if '.npy' not in pathi:
        continue
    data = np.load(os.path.join(root, pathi))
    data[data > 0] = 1
    plt.imshow(data)
    plt.show()
    input('pause')


