import os

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def P10(root):
    print('process10')
    save_root = os.path.join(root, 'p10_result')
    if not os.path.exists(save_root):
        os.makedirs(save_root)

    agg_root = os.path.join(root, 'p6_2corr_result_smooth/MAP')
    FLI_root = os.path.join(root, 'p7coor_result/MAP')
    FLII_root = os.path.join(root, 'p8coor_result/MAP')

    agg_paths = os.listdir(agg_root)
    agg_paths.sort()
    FLI_paths = os.listdir(FLI_root)
    FLI_paths.sort()
    FLII_paths = os.listdir(FLII_root)
    FLII_paths.sort()

    flag = np.array([
        [0, 255, 255],
        [255, 0, 0],
        [255, 215, 0]
    ])

    for i in range(len(agg_paths)):
        pathi = agg_paths[i]
        if '.npy' not in pathi:
            continue
        print(pathi)
        MAP = np.load(os.path.join(agg_root, pathi))
        img = np.zeros([MAP.shape[0], MAP.shape[1], 3])
        img = img.astype(np.uint8)

        indx = (MAP > 0)
        img[indx, :] = flag[0]
        ''''''
        if pathi in FLI_paths:
            MAP = np.load(os.path.join(FLI_root, pathi))
            indx = (MAP > 0)
            img[indx, :] = flag[1]
            
            if pathi in FLII_paths:
                MAP = np.load(os.path.join(FLII_root, pathi))
                indx = (MAP > 0)
                img[indx, :] = flag[2]

        plt.imshow(img), plt.axis('off')
        # plt.show()
        # input('pause')
        np.save(os.path.join(save_root, pathi), img)
        cv2.imwrite(os.path.join(save_root, pathi.replace('npy', 'png')), img[:, :, ::-1])


if __name__ == '__main__':
    P10(root='1329687')
