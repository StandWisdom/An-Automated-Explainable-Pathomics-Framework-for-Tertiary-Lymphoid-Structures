import os
import cv2
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def expand(coor, pix, rate, lenth):
    expansion_pixels = lenth / (pix * rate)
    size = coor.max() + 1000
    mask = np.zeros([size, size]).astype(np.uint8)
    cv2.drawContours(mask, [coor], -1, 255, -1)
    kernel_size = int(2 * expansion_pixels + 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

    image1 = cv2.dilate(mask, kernel, iterations=1).astype(np.uint8)
    contours1, _ = cv2.findContours(image1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    image2 = cv2.erode(mask, kernel, iterations=1).astype(np.uint8)
    contours2, _ = cv2.findContours(image2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # mask_copy = np.zeros([size, size])
    # cv2.drawContours(mask_copy, [coor], -1, 255, 10)
    # cv2.drawContours(mask_copy, contours1, -1, 255, 10)
    # cv2.drawContours(mask_copy, contours2, -1, 255, 10)
    # plt.imshow(mask_copy)
    # plt.show()
    return contours1[0], contours2[0]

roots = '/data/xtw/Project/data/coor'
nl = ['1_10x', '2_20x', '3_20x']
czi_roots = ['/data/xtw/Project/pythonProject/00key_hist_0.9_10X',
            '/data/xtw/Project/pythonProject/00key_hist_0.9',
            '/data/xtw/Project/pythonProject/00key_hist_0.9']
for t in range(len(nl)):
    root = os.path.join(roots, nl[t])
    paths = os.listdir(root)
    paths.sort()
    print(len(paths))

    czi_root = czi_roots[t]

    if t == 0:
        rate = 10
        pix = 0.3454
    else:
        rate = 20
        pix = 0.1721
    lenth = 500
    for i in range(len(paths)):
        pathi = os.path.join(root, paths[i])
        czi_pathi = os.path.join(czi_root, paths[i].split('_')[0])
        print(pathi)
        filename = os.path.basename(pathi).split('_')[0]
        save_root = os.path.join(nl[t], filename, 'distance')
        if not os.path.exists(save_root):
            os.makedirs(save_root)
        # input('pause')

        data = np.load(pathi, allow_pickle=True).item()
        coor = data['coor']
        if len(coor) != 2:
            coor = (np.array(coor) / rate).astype(int)
            coor = np.expand_dims(coor, 1)
        if len(coor) == 2:
            COOR = []
            for j in range(len(coor)):
                coorj = (np.array(coor[j]) / rate).astype(int)
                coorj = np.expand_dims(coorj, 1)
                COOR.append(coorj)
            coor = np.vstack([COOR[0], COOR[1]])
        coor_d, coor_e = expand(coor, pix=pix, rate=rate, lenth=lenth)

        coor_root = os.path.join(czi_pathi, 'p6_3coor_correct')
        coor_paths = os.listdir(coor_root)
        coor_paths.sort()

        Info = []
        for j in range(len(coor_paths)):
            coor_pathj = os.path.join(coor_root, coor_paths[j])
            basename = os.path.basename(coor_pathj)
            indx = int(os.path.splitext(basename)[0].split('_')[-1])
            print(indx, basename)

            czi_coorj = np.load(coor_pathj)
            M = cv2.moments(czi_coorj)
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            # cv2.drawContours(mask, [czi_coorj], -1, 128, -1)  # draw
            # mask[cy:cy+30, cx:cx+30] = 255  # draw

            dis = abs(coor - [cx, cy]).squeeze(1)
            dis = np.sqrt((dis ** 2)[:, 0] + (dis ** 2)[:, 1])
            D = np.min(dis)
            D = D * pix * rate

            dis1 = abs(coor_d - [cx, cy]).squeeze(1)
            dis1 = np.sqrt((dis1 ** 2)[:, 0] + (dis1 ** 2)[:, 1])
            D1 = np.min(dis1)
            D1 = D1 * pix * rate

            dis2 = abs(coor_e - [cx, cy]).squeeze(1)
            dis2 = np.sqrt((dis2 ** 2)[:, 0] + (dis2 ** 2)[:, 1])
            D2 = np.min(dis2)
            D2 = D2 * pix * rate

            line = [indx, D, D1, D2]
            Info.append(line)

        df = pd.DataFrame(Info)
        df.columns = ['TLS index', 'mid', 'large', 'small']
        df.to_csv(os.path.join(save_root, 'distance.csv'))
        print(df)
        # input('pause')
