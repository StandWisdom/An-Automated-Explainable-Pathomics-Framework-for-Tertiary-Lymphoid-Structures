import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


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


def P6_smooth(root, ksize=(9, 9), open_iter=5, dilate_iter=1):
    print('process6 smooth')
    save_root = os.path.join(root, 'p6_2corr_result_smooth')
    save_mask = os.path.join(save_root, 'MAP')
    save_value = os.path.join(save_root, 'VALUE')
    save_coor = os.path.join(save_root, 'coor')
    if not os.path.exists(save_mask):
        os.makedirs(save_mask)
    if not os.path.exists(save_value):
        os.makedirs(save_value)
    if not os.path.exists(save_coor):
        os.makedirs(save_coor)

    mask_root = os.path.join(root, 'p6_2corr_result/MAP')
    value_root = os.path.join(root, 'p6_2corr_result/VALUE')
    coor_root = os.path.join(root, 'p6_2corr_result/coor')
    paths = os.listdir(mask_root)
    paths.sort()

    coor_list = []
    for i in range(len(paths)):
        pathi = paths[i]
        if '.npy' not in pathi:
            continue
        indx = pathi.split('_')[0]
        print(pathi)

        mask = np.load(os.path.join(mask_root, pathi))
        coor = np.load(os.path.join(coor_root, 'coor_' + indx + '.npy'))
        value = np.load(os.path.join(value_root, pathi))

        temp = np.zeros_like(mask)
        cv2.drawContours(temp, [coor], -1, 1, -1)
        img = temp.astype(np.uint8)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, ksize)
        opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel, iterations=open_iter)
        if np.sum(opening > 0) == 0:
            continue
        opening = cv2.dilate(opening, kernel, iterations=dilate_iter)
        max_mask_list, smooth_coor_list = FindRegion(opening)
        for t in range(len(max_mask_list)):
            max_mask = max_mask_list[t]
            smooth_coor = smooth_coor_list[t]
            smooth_mask = max_mask * mask
            smooth_value = max_mask * value

            IDlist = set(np.reshape(smooth_mask, [1, -1])[0])
            num = len(IDlist) - 1
            if num < 100:
                continue

            if t != 0:
                pathi = pathi[:3].replace(indx, str(100 + int(indx))) + pathi[3:]
                indx = str(100 + int(indx))

            np.save(os.path.join(save_mask, pathi), smooth_mask)
            cv2.imwrite(os.path.join(save_mask, pathi.replace('.npy', '.png')), smooth_mask)
            np.save(os.path.join(save_value, pathi), smooth_value)
            cv2.imwrite(os.path.join(save_value, pathi.replace('.npy', '.png')), smooth_value)
            np.save(os.path.join(save_coor, 'coor_' + indx + '.npy'), smooth_coor)
            # plt.figure(1), plt.imshow(max_mask)
            # plt.figure(2), plt.imshow(smooth_mask)
            # plt.figure(3), plt.imshow(smooth_value)
            # plt.show()
            # input('pause')

            ID, newx, newy, _ = pathi.split('_')
            ID = int(ID)
            newx = int(newx.split('x')[-1])
            newy = int(newy.split('y')[-1])
            shape = smooth_mask.shape
            name = pathi.replace('.npy', '')
            line = [name, num, ID, newx, newy, shape]
            coor_list.append(line)
            # input('pause')
    df = pd.DataFrame(coor_list)
    df.columns = ['name', 'count', 'index', 'newx', 'newy', 'shape']
    print(df)
    csv_root = os.path.join(root, 'p6_2corr_result_smooth/CDcomb.csv')
    df.to_csv(csv_root)


if __name__ == '__main__':
    P6_smooth(root='1567648')

