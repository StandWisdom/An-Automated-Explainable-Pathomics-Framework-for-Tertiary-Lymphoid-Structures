import os
import scipy
import slideio
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from _1czi_save import czi_load


def hist_equalize(img, mx=16400):
    # 计算直方图
    hist, bins = np.histogram(img.flatten(), bins=mx, range=[0, mx])
    # peaks, _ = scipy.signal.find_peaks(hist, 10**6)
    peaks, _ = scipy.signal.find_peaks(hist, 255)
    results_full = scipy.signal.peak_widths(hist, peaks, rel_height=0.999)
    right_ips = results_full[-1]
    max_ips = right_ips.max()
    print(max_ips)
    hist[:int(max_ips+1)] = 0
    # 计算累积分布函数
    cdf = hist.cumsum()
    # 归一化累积分布函数
    cdf_normalized = (cdf - cdf.min()) * mx / (cdf.max() - cdf.min())
    # cdf_normalized = cdf * hist.max() / cdf.max()
    # 使用累积分布函数进行直方图均衡化
    img[img < int(max_ips+1)] = 0
    equalized_img = np.interp(img.flatten(), bins[:-1], cdf_normalized)
    equalized_img = equalized_img.reshape(img.shape)

    return equalized_img.astype(np.uint16)


def read_block(scene, thum, channel, threshold, show_flag=False):
    rect = scene.rect
    size = (0, int(rect[3] / thum))
    block = scene.read_block(rect, size, [channel])
    block = hist_equalize(block)
    block[block < threshold] = 0

    if show_flag:
        plt.imshow(block)
        plt.show()
        cv2.imwrite('block.jpg', block)
        print(threshold)
        input('pause')

    return block


def coor_show(img, src_coor):
    temp = int(img.max()/2)
    for i in range(len(src_coor)):
        up, down, left, right = src_coor[i]
        img = cv2.rectangle(img, (left, up), (right, down), (temp, 0, 0), thickness=5)
    plt.imshow(img)
    plt.show()


def coor_cut(img, src_coor, save_path):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    for i in range(len(src_coor)):
        print(i)
        up, down, left, right = src_coor[i]
        tile = img[up:down, left:right]
        np.save(os.path.join(save_path, '%d_agg.npy' % i), tile)
        # cv2.imwrite(os.path.join(save_path, '%d_agg.png' % i), tile)


def P3_hist(root, czi_path, channel, thum, threshold, edge, show_flag=False):
    print('process3: need czi_path')
    celist = ['DAPI', 'CD20', 'CD3', 'CD23', 'CD21']
    ty = celist[channel]
    print(ty)
    scene = czi_load(czi_path)
    block = read_block(scene, thum=thum, channel=channel, threshold=threshold, show_flag=show_flag)
    print(block.shape, block.max())

    save_path = os.path.join(root, 'p3channel/' + ty)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    coor_root = os.path.join(root, 'p2result/coor')
    paths = os.listdir(coor_root)
    num = len(paths)
    rate = 20 / thum
    List = []
    for i in range(num):
        path = os.path.join(coor_root, 'coor_%d.npy' % i)
        # print(path)
        coor = np.load(path)
        D_coor = rate * coor
        List.append(D_coor)

        x_min, x_max = D_coor[:, 0, 0].min(), D_coor[:, 0, 0].max()
        y_min, y_max = D_coor[:, 0, 1].min(), D_coor[:, 0, 1].max()

        new_y_min, new_y_max = int(y_min - edge), int(y_max + edge)
        new_x_min, new_x_max = int(x_min - edge), int(x_max + edge)

        TLSi = block[new_y_min:new_y_max, new_x_min: new_x_max]

        area_mask = np.zeros_like(TLSi)
        D_coor[:, 0, 0] = D_coor[:, 0, 0] - x_min
        D_coor[:, 0, 1] = D_coor[:, 0, 1] - y_min
        D_coor = D_coor.astype(int)
        cv2.drawContours(area_mask, [D_coor], -1, 255, -1)

        area_mask[area_mask == 255] = 1
        # TLSi = TLSi * area_mask

        np.save(os.path.join(save_path, '%d_x%d_y%d_img.npy' % (i, new_x_min, new_y_min)), TLSi)
        cv2.imwrite(os.path.join(save_path, '%d_x%d_y%d_img.png' % (i, new_x_min, new_y_min)), TLSi)


if __name__ == '__main__':
    # czi_path = '/data/xtw/Project/data/TLS/0105-AXTC177-P2-1-T5-3-20X-1406542-B26.czi'
    # P3_hist('1406542', czi_path, channel=4, thum=4, threshold=16400*0.85, edge=50, show_flag=True)
    img = cv2.imread(r'C:\Users\XTW\Desktop\hist.png')
    img_h = hist_equalize(img, 255)
    cv2.imwrite('ori.png', img)
    cv2.imwrite('hist.png', img_h.astype(np.uint8))


