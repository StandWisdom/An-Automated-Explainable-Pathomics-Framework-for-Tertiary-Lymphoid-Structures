import os
import slideio
import numpy as np
import pandas as pd

import cv2
import matplotlib.pyplot as plt
from _1czi_save import czi_load


def read_block(scene, thum, channel, thl, thh, percent, show_flag=False):
    rect = scene.rect
    size = (0, int(rect[3] / thum))
    block = scene.read_block(rect, size, [channel])
    if percent:
        max_glob = block.max()
        print(max_glob)
        if channel == 2 and (max_glob*thl > 600):
            print('channel 2 force threshold')
            thl = 600 / max_glob
        if channel == 3 and (max_glob*thl > 1200):
            print('channel 3 force threshold')
            thl = 1200 / max_glob
        if channel == 4 and (max_glob*thl > 1000):
            print('channel 4 force threshold')
            thl = 1000 / max_glob
        block = block / max_glob
    block[block < thl] = 0
    if thh != 0:
        block[block > thh] = 0

    if show_flag:
        plt.imshow(block)
        plt.show()
    if percent:
        block = (max_glob*block).astype(np.uint16)
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


def P3(root, czi_path, channel, thum, thl, thh, percent=False, show_flag=False):
    if percent:
        assert thl < 1 and thh < 1, 'thl and thh should be percent'
    print('process3: need czi_path')

    celist = ['DAPI', 'CD20', 'CD3', 'CD23', 'CD21']
    ty = celist[channel]
    print(ty)
    scene = czi_load(czi_path)
    block = read_block(scene, thum=thum, channel=channel, thl=thl, thh=thh,
                       percent=percent, show_flag=show_flag)
    print(block.shape)
    # plt.imshow(block)
    # plt.show()

    save_path = os.path.join(root, 'p3channel/' + ty)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    coor_root = os.path.join(root, 'p2result/coor')
    paths = os.listdir(coor_root)
    num = len(paths)
    rate = 20 / thum
    edge = 50
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
        new_x_min, new_y_min = int(max(0, new_x_min)), int(max(0, new_y_min))

        TLSi = block[new_y_min:new_y_max, new_x_min: new_x_max]

        area_mask = np.zeros_like(TLSi)
        D_coor[:, 0, 0] = D_coor[:, 0, 0] - x_min
        D_coor[:, 0, 1] = D_coor[:, 0, 1] - y_min
        D_coor = D_coor.astype(int)
        cv2.drawContours(area_mask, [D_coor], -1, 255, -1)

        area_mask[area_mask == 255] = 1
        # TLSi = TLSi * area_mask
        print('%d_x%d_y%d_img.npy' % (i, new_x_min, new_y_min))

        np.save(os.path.join(save_path, '%d_x%d_y%d_img.npy' % (i, new_x_min, new_y_min)), TLSi)
        cv2.imwrite(os.path.join(save_path, '%d_x%d_y%d_img.png' % (i, new_x_min, new_y_min)), TLSi)


if __name__ == '__main__':
    czi_path = '/data/xtw/Project/data/TLS/new data 20240115/mxIF/0105-AXTC177-P2-1-T5-3-20X-1406542-B26.czi'
    P3('1406542', czi_path, channel=4, thum=4, thl=0, thh=0)
