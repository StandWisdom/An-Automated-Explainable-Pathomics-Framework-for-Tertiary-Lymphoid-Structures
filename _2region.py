import os.path

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def seg_threshold(img):
    img = 255*img/img.max()
    img = img.astype(np.uint8)
    # img = cv2.medianBlur(img, 9)

    img[img > 0] = 255

    mask = img.astype(np.uint8)

    element = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    mask = cv2.dilate(mask, element, iterations=2)  # 保留整块区域
    # cv2.namedWindow('mask1', 0)
    # cv2.resizeWindow('mask1', 700, 700)  # 自己设定窗口图片的大小
    # cv2.imshow('mask1', mask)
    # cv2.waitKey()
    # cv2.destroyAllWindows()

    mask = cv2.erode(mask, element, iterations=2)  # 去除噪点
    # cv2.namedWindow('mask2', 0)
    # cv2.resizeWindow('mask2', 700, 700)
    # cv2.imshow('mask2', mask)
    # cv2.waitKey()
    # cv2.destroyAllWindows()

    return mask


def FindMaxRegion(mask):
    area_list = []
    contour_list = []

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        area = cv2.contourArea(contour)
        area_list.append(area)
        contour_list.append(contour)

    index = np.argsort(area_list)
    LIST = []
    num = len(area_list) if len(area_list) < 50 else 50
    for i in range(1, num+1):
        temp = contour_list[index[-i]]
        LIST.append(temp)

    area_mask = np.zeros_like(mask)
    cv2.drawContours(area_mask, LIST, -1, 255, -1)

    return area_mask, LIST


def P2(root):
    print('process2: thum/CD20/img.npy')
    paths = os.listdir(root)
    for i in paths:
        if 'thum' in i:
            path = i
        else:
            continue
    path = os.path.join(root, path, 'CD20', 'img.npy')
    img = np.load(path)
    mask = seg_threshold(img)
    mask, List = FindMaxRegion(mask)  # find region

    save_root = os.path.join(root, 'p2result')
    save_path = os.path.join(save_root, 'mask')
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    cv2.imwrite(os.path.join(save_path, 'mask.png'), mask)
    np.save(os.path.join(save_path, 'mask.npy'), mask)

    save_path = os.path.join(save_root, 'coor')
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    for i in range(len(List)):
        data = List[i]
        # print(data.shape)
        np.save(os.path.join(save_path, 'coor_%d.npy' % i), data)


if __name__ == '__main__':
    P2(root='1406542')
