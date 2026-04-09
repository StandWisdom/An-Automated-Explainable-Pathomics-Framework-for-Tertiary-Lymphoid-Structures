import os
import cv2
import numpy as np
import pandas as pd
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


mask_path = '/data/xtw/Project/pythonproject/00one_clik/1406542/p6_2corr_result/MAP/10_x5325_y6965_img.npy'
value_path = '/data/xtw/Project/pythonproject/00one_clik/1406542/p6_1coor_result/VALUE/10_x5325_y6965_img.npy'

mask = np.load(mask_path)
plt.imshow(mask), plt.show()
dst = cv2.blur(mask.astype(np.uint8), (5, 5))
plt.imshow(dst)
plt.show()

value = np.load(value_path)
value_copy = np.copy(value)
value[value > 0] = 1
plt.imshow(value), plt.show()

mask = mask*value
plt.imshow(mask), plt.show()

value_copy = (255*(value_copy/value_copy.max())).astype(np.uint8)
are_mask, coor = FindMaxRegion(value_copy)
plt.imshow(are_mask)
plt.show()
