import os
import cv2
import matplotlib.pyplot as plt
import slideio
import numpy as np
from _1czi_save import czi_load
from _3czi_hist_cut import hist_equalize


def read_block(scene, thum, channel, threshold, show_flag=False):
    rect = scene.rect
    size = (0, int(rect[3] / thum))
    block = scene.read_block(rect, size, [channel])
    if channel == 0:
        max_glob = block.max()
        print(max_glob)
        block = block / max_glob
        block[block < threshold] = 0
        block = (max_glob * block).astype(np.uint16)
    else:
        block = hist_equalize(block)
        block[block < threshold] = 0
    if show_flag:
        plt.imshow(block)
        plt.show()
    return block


def P12(root, czi_path, channel, thum, threshold, show_flag):
    celist = ['DAPI', 'CD20', 'CD3', 'CD23', 'CD21', 'PANCK', 'KI67']
    ty = celist[channel]
    print(ty)

    scene = czi_load(czi_path)
    block = read_block(scene, thum=thum, channel=channel, threshold=threshold, show_flag=show_flag)
    print(block.shape)

    coor_root = os.path.join(root, 'p6_3coor_correct')
    coor_paths = os.listdir(coor_root)
    coor_paths.sort()

    names_root = os.path.join(root, 'p6_2corr_result_smooth/MAP')
    names = os.listdir(names_root)
    names.sort()
    names = [x for x in names if 'png' not in x]

    coor_paths_copy = [os.path.splitext(x)[0].split('_')[-1] for x in coor_paths]
    names_copy = [os.path.splitext(x)[0].split('_')[0] for x in names]
    INDEX = []
    for x in coor_paths_copy:
        INDEX.append(names_copy.index(x))
    print(len(INDEX), INDEX)

    save_root = os.path.join(root, 'p12_addition')
    rlist = ['rate1', 'rate2', 'rate3']
    for i in range(len(rlist)):
        if not os.path.exists(os.path.join(save_root, rlist[i], ty)):
            os.makedirs(os.path.join(save_root, rlist[i], ty))

    for i in range(len(coor_paths)):
        coor_pathi = coor_paths[i]
        print(coor_pathi)
        save_name = names[INDEX[i]]
        print(save_name)
        x, y = save_name.split('_')[1:3]
        x, y = int(x[1:]), int(y[1:])
        position = [x, y]
        print('posision:', position)

        coor = np.load(os.path.join(coor_root, coor_pathi))
        coor = np.squeeze(coor, axis=1)
        coor = 5 * coor

        mu = cv2.moments(coor, False)
        cx, cy = round(mu['m10'] / mu['m00']), round(mu['m01'] / mu['m00'])
        print('cx, cy:', cx, cy)
        L = abs(coor - [cx, cy])
        dis = np.sum(L, axis=1)
        indx = np.argmax(dis)
        [a, b] = L[indx]
        l = round(np.sqrt(a ** 2 + b ** 2))
        print('l:', l)

        mc = [cx - x, cy - y]  # mask central
        print('mc:', mc)
        MAP = np.load(os.path.join(names_root, save_name))
        coor = coor - position
        mask = np.zeros_like(MAP, dtype=np.uint8)
        cv2.drawContours(mask, [coor], -1, 255, -1)
        # plt.imshow(mask)
        # plt.show()
        for j in range(1, len(rlist)+1):
            path = os.path.join(save_root, rlist[j - 1], ty)
            print(path)

            print('block shape', block.shape)
            print('position', cy-j*l, cy+j*l, cx-j*l, cx+j*l)
            a, b, c, d = cy-j*l, cy+j*l, cx-j*l, cx+j*l
            pa, pc, pb, pd = 0, 0, 0, 0
            if a < 0:
                pa = -a
                a = 0
            if c < 0:
                pc = -c
                c = 0
            if b > block.shape[0]:
                pb = b - block.shape[0]
                b = block.shape[0]
            if d > block.shape[1]:
                pd = d - block.shape[1]
                d = block.shape[1]
            img = block[a:b, c:d]  # value
            np.save(os.path.join(path, save_name), img)
            print(img.shape, mask.shape)
            cv2.imwrite(os.path.join(path, save_name.replace('npy', 'png')), img)
            # input('pause')

            MASK = np.zeros_like(img, dtype=np.uint8)
            if ((j * l - pa) - mc[1]) > 0:
                a = (j * l - pa) - mc[1]
                aa = mc[1] - mc[1]
            else:
                a = (j * l - pa) - (j * l - pa)
                aa = mc[1] - (j * l - pa)
            if (mask.shape[0] - mc[1]) <= (j * l - pb):
                b = (j * l - pa) + (mask.shape[0] - mc[1])
                bb = mc[1] + (mask.shape[0] - mc[1])
            else:
                b = (j * l - pa) + (j * l - pb)
                bb = mc[1] + (j * l - pb)

            if ((j * l - pc) - mc[0]) > 0:
                c = (j * l - pc) - mc[0]
                cc = mc[0] - mc[0]
            else:
                c = (j * l - pc) - (j * l - pc)
                cc = mc[0] - (j * l - pc)
            if (mask.shape[1] - mc[0]) <= (j * l - pd):
                d = (j * l - pc) + (mask.shape[1] - mc[0])
                dd = mc[0] + (mask.shape[1] - mc[0])
            else:
                d = (j * l - pc) + (j * l - pd)
                dd = mc[0] + (j * l - pd)
            MASK[a:b, c:d] = mask[aa:bb, cc:dd]
            # MASK[j * l - mc[1]:j * l + (mask.shape[0] - mc[1]), j * l - mc[0]:j * l + (mask.shape[1] - mc[0])] = mask
            # plt.imshow(img)
            # plt.show()
            # plt.imshow(MASK)
            # plt.show()
            # input('pause')
            cv2.imwrite(os.path.join(path, save_name.replace('.npy', '_mask.png')), MASK)
            # input('pause')


if __name__ == '__main__':
    pmax = 16400
    czi_path = '/nfs/data351/xtw/czi/0617-AXTC177-P3-1-T5-5-1567648-B6.czi'
    # P12('1406542', czi_path, channel=0, thum=4, threshold=0.1, show_flag=False)
    P12('1567648', czi_path, channel=5, thum=4, threshold=pmax * 0.5, show_flag=False)
    P12('1567648', czi_path, channel=6, thum=4, threshold=pmax * 0.5, show_flag=False)

