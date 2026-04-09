import os

import slideio
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from _9show import merge


def P9_tr(root):
    colormap = np.array([
        [0, 0, 0],
        [0, 255, 255],
        [0, 255, 0],
        [255, 128, 0],
        [255, 0, 0]
    ])
    # plt.figure(0), plt.imshow(np.expand_dims(colormap, 0)), plt.axis('off')

    save_root = 'p9_result'
    if not os.path.exists(save_root):
        os.makedirs(save_root)

    hsvimg = np.load(os.path.join(save_root, 'dataimg.npy'))
    print(hsvimg.shape)
    # plt.imshow(hsvimg)
    # plt.show()

    MAP_root = os.path.join(root, 'p6coor_result/MAP')
    paths = os.listdir(MAP_root)
    show_map = np.zeros_like(hsvimg)
    coor_list = []
    for i in range(len(paths)):
        pathi = paths[i]
        if '.npy' not in pathi:
            continue
        name = os.path.splitext(pathi)[0]
        # print(name)
        indx, x, y, _ = pathi.split('_')
        indx = int(indx)
        x, y = int(x[1:])//5, int(y[1:])//5
        MAP = np.load(os.path.join(MAP_root, pathi))
        MAP[MAP > 0] = 255
        MAP = MAP.astype(np.uint8)

        reshape = cv2.resize(MAP, (int(MAP.shape[0]/5), int(MAP.shape[1]/5)), interpolation=cv2.INTER_NEAREST)
        reshape[reshape != 0] = 1
        leny, lenx = reshape.shape

        show_map[y:y + leny, x:x + lenx, :] = hsvimg[y:y+leny, x:x+lenx, :] * np.dstack([reshape, reshape, reshape])
        coor_list.append([indx, name, y, y + leny, x, x + lenx])

    df = pd.DataFrame(coor_list)
    df.columns = ['index', 'name', 'y', 'y+leny', 'x', 'x+lenx']
    print(df)

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

    img = show_map
    for line in df.iloc:
        lva = line.values
        idx = lva[0]
        if idx in fli_list:
            cr = (0, 0, 255)
        elif idx in flii_list:
            cr = (255, 0, 255)
        else:
            cr = (255, 255, 255)

        y1, y2, x1, x2 = lva[2:6]
        img = cv2.rectangle(img, pt1=(x1, y1), pt2=(x2, y2), color=cr, thickness=4)
        print(lva)
    cv2.imwrite(os.path.join(save_root, 'img_trangle.png'), img[:, :, ::-1])
    plt.imshow(img)
    plt.show()


if __name__ == '__main__':
    P9_tr('1406542')

