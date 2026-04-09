import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

mask_path = '/data/xtw/Project/pythonProject/00one_clik/1454807/p6corr_result/MAP/0_x12835_y20535_img.npy'
file_path = '/data/xtw/Project/pythonProject/00one_clik/1454807/p6_express_result/0_x12835_y20535_img.csv'

mask = np.load(mask_path)
file = pd.read_csv(file_path, index_col=0)

CD20list = file[file['CD20'] == 1]['cell_id'].values
CD3list = file[file['CD3'] == 1]['cell_id'].values

print(CD20list)
print(CD3list)
'''
x, y = mask.shape
flag = np.zeros_like(mask)
for i in range(x):
    for j in range(y):
        if mask[i, j] in CD20list:
            flag[i, j] = 255
        if mask[i, j] in CD3list:
            flag[i, j] = 128
'''
save_path = 'add_result'
if not os.path.exists(save_path):
    os.makedirs(save_path)
# np.save(os.path.join(save_path, 'flag.npy'), flag)

flag = np.load(os.path.join(save_path, 'flag.npy'))

color = np.zeros([mask.shape[0], mask.shape[1], 3])
color[flag == 128, :] = [0, 255, 0]
color[flag == 255, :] = [0, 128, 255]
color = color.astype(np.uint8)

plt.imshow(color), plt.axis('off')
plt.show()


'''
import numpy as np
import matplotlib.pyplot as plt

mask_path = '/data/xtw/Project/pythonProject/00one_clik/1454807/p6corr_result/MAP/0_x12835_y20535_img.npy'
valu_path = '/data/xtw/Project/pythonProject/00one_clik/1454807/p3channel/CD20/0_x12835_y20535_img.npy'

mask = np.load(mask_path)
valu = np.load(valu_path)

valu[valu > 0] = 1
mask = mask * valu

mask[mask > 0] = 255
color = np.zeros([mask.shape[0], mask.shape[1], 3])
color[mask == 255, :] = [0, 128, 255]

plt.imshow(color.astype(np.uint8)), plt.axis('off')
plt.show()

'''

