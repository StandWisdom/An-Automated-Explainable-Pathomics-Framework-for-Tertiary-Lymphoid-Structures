import os
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt


def FindMaxRegion(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    max_area = 0
    max_contour = None
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > max_area:
            max_area = area
            max_contour = contour

    max_area_mask = np.zeros_like(mask)
    cv2.drawContours(max_area_mask, [max_contour], -1, 1, -1)
    return max_area_mask, max_contour


def FindRegion(mask):
    area_list = []
    contour_list = []

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        area = cv2.contourArea(contour)
        area_list.append(area)
        contour_list.append(contour)

    index = np.argsort(area_list)
    LIST = []
    mask_list = []
    num = len(area_list) if len(area_list) < 2 else 2
    for i in range(1, num+1):
        temp = contour_list[index[-i]]
        area_mask = np.zeros_like(mask)
        cv2.drawContours(area_mask, [temp], -1, 255, -1)
        LIST.append(temp)
        mask_list.append(area_mask)

    return mask_list, LIST


def P6_corr(root):
    print('process6 corr')
    save_root = os.path.join(root, 'p6_2corr_result')
    coor_save_path = os.path.join(save_root, 'coor')
    map_save_path = os.path.join(save_root, 'MAP')
    value_save_path = os.path.join(save_root, 'VALUE')
    if not os.path.exists(coor_save_path):
        os.makedirs(coor_save_path)
    if not os.path.exists(map_save_path):
        os.makedirs(map_save_path)
    if not os.path.exists(value_save_path):
        os.makedirs(value_save_path)

    read_root = os.path.join(root, 'p6_1coor_result/MAP')
    paths = os.listdir(read_root)
    paths.sort()
    for i in range(len(paths)):
        pathi = paths[i]
        if '.npy' not in pathi:
            continue
        print(pathi)
        indx = int(pathi.split('_')[0])
        maski = np.load(os.path.join(read_root, pathi))
        valuei = np.load(os.path.join(read_root.replace('MAP', 'VALUE'), pathi))
        maski_save = np.copy(maski)
        maski[maski > 0] = 255
        maski = maski.astype(np.uint8)
        are_mask_list, coor_list = FindRegion(maski)
        for t in range(len(are_mask_list)):
            maski_copy = np.copy(maski_save)
            are_mask = are_mask_list[t]
            coor = coor_list[t]
            maski_copy = maski_copy * are_mask
            value_save = valuei * are_mask

            SET = set(np.reshape(maski_copy, (1, -1))[0])
            num = len(SET) - 1
            if num < 100:
                continue

            if t != 0:
                pathi = pathi[:3].replace(str(indx), str(50 + indx)) + pathi[3:]
                indx = 50 + indx
            print(indx, pathi)

            np.save(os.path.join(coor_save_path, 'coor_%d' % indx), coor)
            np.save(os.path.join(map_save_path, pathi), maski_copy)
            maski_copy[maski_copy > 0] = 255
            cv2.imwrite(os.path.join(map_save_path, pathi.replace('.npy', '.png')), maski_copy)

            np.save(os.path.join(value_save_path, pathi), value_save)
            plt.imshow(value_save), plt.axis('off')
            plt.savefig(os.path.join(value_save_path, pathi.replace('npy', 'png')))


if __name__ == '__main__':
    P6_corr(root='/data/xtw/Project/pythonProject/00key_hist_0.9/1567648')
