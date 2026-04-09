import os
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from _1czi_save import czi_load


def merge(img, colormap):
    hsvmap = cv2.cvtColor(np.expand_dims(colormap, axis=0).astype(np.uint8), cv2.COLOR_RGB2HSV)
    hsvmap = hsvmap.squeeze(0)
    x, y = np.shape(img)[0], np.shape(img)[1]
    color = np.zeros((x, y, 3))
    for p in range(x):
        for q in range(y):
            remap = np.copy(hsvmap)
            rate = np.clip(img[p, q, :] / 2500, 0, 1)
            remap[1:, 2] = remap[1:, 2] * rate
            remap = cv2.cvtColor(np.expand_dims(remap, axis=0).astype(np.uint8), cv2.COLOR_HSV2RGB)
            remap = remap.squeeze(0)
            color[p, q, :] = np.max(remap, axis=0)
    # color = color.astype(np.uint8)

    return color


def P9(root, czi_path):
    print('process9: need czi_path')
    colormap = np.array([
        [0, 0, 0],
        [0, 255, 255],
        [0, 255, 0],
        [255, 128, 0],
        [255, 0, 0]
    ])
    # plt.figure(0), plt.imshow(np.expand_dims(colormap, 0)), plt.axis('off')

    flag = np.array([
        [255, 255, 255],
        [0, 0, 255],
        [255, 0, 255]
    ])

    save_root = os.path.join(root, 'p9_result')
    if not os.path.exists(save_root):
        os.makedirs(save_root)

    ''''''
    thum = 20
    scene = czi_load(czi_path)
    rect = scene.rect
    size = (0, int(rect[3] / thum))
    block = scene.read_block(rect, size, [1, 2, 3, 4])
    for i in range(block.shape[-1]):
        block[block[:, :, i] < 1000, i] = 0
        block[block[:, :, i] > 9000, i] = 0
    np.save(os.path.join(save_root, 'data.npy'), block)
    block = np.load(os.path.join(save_root, 'data.npy'))
    hsvimg = merge(block, colormap)
    np.save(os.path.join(save_root, 'dataimg.npy'), hsvimg)
    cv2.imwrite(os.path.join(save_root, 'dataimg.png'), hsvimg[:, :, ::-1])

    hsvimg = np.load(os.path.join(save_root, 'dataimg.npy'))
    print(hsvimg.shape)

    agg_path = os.path.join(root, 'p6_1coor_result/coor/CDcomb.csv')
    FLI_path = os.path.join(root, 'p7coor_result/coor/CD21.csv')
    FLII_path = os.path.join(root, 'p8coor_result/coor/CD23.csv')

    FLII = pd.read_csv(FLII_path, index_col=0)['name'].values
    flii_list = []
    for flii in FLII:
        inx = int(flii.split('_')[0])
        flii_list.append(inx)

    FLI = pd.read_csv(FLI_path, index_col=0)['name'].values
    fli_list = []
    for fli in FLI:
        inx = int(fli.split('_')[0])
        if inx not in flii_list:
            fli_list.append(inx)

    AGG = pd.read_csv(agg_path, index_col=0)['name'].values
    Agg_list = []
    for agg in AGG:
        inx = int(agg.split('_')[0])
        if (inx not in flii_list) and (inx not in fli_list):
            Agg_list.append(inx)

    TLS_num = len(AGG)
    record = np.zeros([TLS_num, 2]).astype(int)
    for i in range(TLS_num):
        inx = int(AGG[i].split('_')[0])
        record[i, 0] = inx
        if inx in Agg_list:
            record[i, 1] = 0
        elif inx in fli_list:
            record[i, 1] = 1
        elif inx in flii_list:
            record[i, 1] = 2
    df = pd.DataFrame(record)
    df.columns = ['id', 'type']
    df.to_csv(os.path.join(save_root, 'TLS_type.csv'))

    coor_root = os.path.join(root, 'p6_3coor_correct')  # p6_corr
    paths = os.listdir(coor_root)
    paths.sort()

    for i in range(len(paths)):
        pathi = paths[i]
        name = os.path.splitext(pathi)[0]
        inx = int(name.split('_')[-1])
        coor = np.load(os.path.join(coor_root, pathi))
        for line in coor[:, 0, :]:
            if inx in fli_list:
                cr = flag[1]
            elif inx in flii_list:
                cr = flag[2]
            else:
                cr = flag[0]
            hsvimg[line[1] - 2: line[1] + 2, line[0] - 2:line[0] + 2, :] = cr

    area_mask = np.zeros_like(hsvimg)
    for i in range(len(paths)):
        pathi = paths[i]
        name = os.path.splitext(pathi)[0]
        inx = int(name.split('_')[-1])
        coor = np.load(os.path.join(coor_root, pathi))
        cv2.drawContours(area_mask, [coor], -1, [1, 1, 1], -1)
    hsvimg = hsvimg * area_mask

    np.save(os.path.join(save_root, 'coorimg.npy'), hsvimg)
    cv2.imwrite(os.path.join(save_root, 'coorimg.png'), hsvimg[:, :, ::-1])

    plt.figure(1), plt.imshow(hsvimg), plt.axis('off')
    plt.savefig(os.path.join(save_root, 'coorimg2.png'))
    plt.show()


if __name__ == '__main__':
    czi_path = '/data/xtw/Project/data/TLS/0105-AXTC177-P2-1-T5-3-20X-1406542-B26.czi'
    root = '1406542'
    P9(root, czi_path)

