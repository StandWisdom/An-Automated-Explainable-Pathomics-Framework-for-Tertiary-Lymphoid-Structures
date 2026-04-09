import os
import cv2
import numpy as np
import matplotlib.pyplot as plt


def P6_coor_correct(root):
    print('process6 coor correct')
    coor_root = os.path.join(root, 'p6_2corr_result_smooth/coor')  # p6_corr
    coor_paths = os.listdir(coor_root)
    coor_paths.sort()

    map_root = os.path.join(root, 'p6_2corr_result_smooth/MAP')
    map_paths = os.listdir(map_root)
    map_paths.sort()

    save_path = os.path.join(root, 'p6_3coor_correct')
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    for i in range(len(coor_paths)):
        coor_pathi = coor_paths[i]
        print(coor_pathi)
        coor = np.load(os.path.join(coor_root, coor_pathi))
        coor_name = os.path.splitext(coor_pathi)[0]
        coor_indx = int(coor_name.split('_')[-1])

        for map_pathi in map_paths:
            splt = map_pathi.split('_')
            map_indx, x, y = int(splt[0]), splt[1], splt[2]
            if map_indx != coor_indx:
                continue
            else:
                x = int(x[1:])
                y = int(y[1:])
                break
        print(x, y)

        coor[:, 0, 0] = coor[:, 0, 0] + x
        coor[:, 0, 1] = coor[:, 0, 1] + y
        coor = (coor / 5).astype(int)
        np.save(os.path.join(save_path, coor_pathi), coor)


if __name__ == '__main__':
    P6_coor_correct(root='1567648')

