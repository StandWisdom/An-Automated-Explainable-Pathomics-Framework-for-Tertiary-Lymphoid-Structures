import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def P13(root, pix=0.1721, thum=4):
    pix_area = (pix * thum) ** 2

    mid = 'p12_addition'
    rlist = ['rate1', 'rate2', 'rate3']
    chl = ['PANCK', 'KI67']

    save_path = os.path.join(root, 'p13_add_count')
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    for j in range(len(rlist)):  # ['rate1', 'rate2', 'rate3']
        root1 = os.path.join(root, mid, rlist[j], chl[0])
        root2 = root1.replace(chl[0], chl[1])
        print(j, root1)

        path_list = os.listdir(root1)
        path_list.sort()
        # print(path_list)
        idcol, col1, col2 = [], [], []
        for name in path_list:  # TLS
            if '.npy' not in name:
                continue
            path1 = os.path.join(root1, name)
            print(path1)
            mask = cv2.imread(path1.replace('.npy', '_mask.png'))[:, :, 0]
            mask[mask > 0] = 1
            value1 = np.load(path1)
            value1 = value1 * (1 - mask)

            indx = int(name.split('_')[0])
            idcol.append(indx)

            path2 = os.path.join(root2, name)
            value2 = np.load(path2)
            value2 = value2 * (1 - mask)
            value2[value1 == 0] = 0

            num1 = np.sum(value1 > 0) * pix_area
            col1.append(num1)
            num2 = np.sum(value2 > 0) * pix_area if num1 != 0 else 0
            col2.append(num2)
            # input('pause')

        if j == 0:
            df = pd.DataFrame(idcol)
            df.columns = ['TLS index']
        c1 = rlist[j]+chl[0]+'(um2)'
        c2 = rlist[j]+chl[0]+chl[1]+'(um2)'
        df[c1] = col1
        df[c2] = col2

    df.to_csv(os.path.join(save_path, 'count.csv'))
    print(df)


if __name__ == '__main__':
    P13('1543567')

