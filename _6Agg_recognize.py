import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def recgnize(data, CDx):
    data = data-1
    num = data.max()
    # print(num)
    # print(set(data.reshape(1, -1)[0]))
    MAP = np.zeros_like(data)
    VALUE = np.zeros_like(CDx)
    count = 0
    for i in range(1, num+1):
        indx = np.where(data == i)
        coor = np.vstack([indx[0], indx[1]]).T

        map = np.zeros_like(data)
        value = np.zeros_like(CDx)
        for j in range(len(coor)):
            map[coor[j][0], coor[j][1]] = data[coor[j][0], coor[j][1]]
            value[coor[j][0], coor[j][1]] = CDx[coor[j][0], coor[j][1]]
        # plt.figure(0), plt.imshow(map)
        # plt.figure(1), plt.imshow(value)
        # plt.show()
        if np.sum(value > 0) > 0:
            MAP += map
            VALUE += value
            count += 1

    return MAP, VALUE, count


def P6(root):
    print('process6')
    read_root = os.path.join(root, 'p5_cell_result')
    paths = os.listdir(read_root)
    paths.sort()

    # load CD
    CD_root = os.path.join(root, 'p3channel/CD20')
    CD_root2 = CD_root.replace('20', '3')
    CD_paths = os.listdir(CD_root)

    save_root = os.path.join(root, 'p6_1coor_result')
    coor_save_path = os.path.join(save_root, 'coor')
    MAP_save_path = os.path.join(save_root, 'MAP')
    VALUE_save_path = os.path.join(save_root, 'VALUE')
    if not os.path.exists(coor_save_path):
        os.makedirs(coor_save_path)
    if not os.path.exists(MAP_save_path):
        os.makedirs(MAP_save_path)
    if not os.path.exists(VALUE_save_path):
        os.makedirs(VALUE_save_path)

    coor_list = []

    num = len(paths)
    for i in range(num):
        if '.npy' not in paths[i]:
            continue
        # load DAPI
        img = np.load(os.path.join(read_root, paths[i]))
        # load CDx
        for CD_path in CD_paths:
            if CD_path == paths[i]:
                print(CD_path)
                CDx1 = np.load(os.path.join(CD_root, CD_path))
                CDx2 = np.load(os.path.join(CD_root2, CD_path))
                CDx = np.maximum(CDx1, CDx2)
                MAP, VALUE, count = recgnize(img, CDx)
                print(count)

                if count > 100:
                    name = os.path.splitext(paths[i])[0]
                    index, newx, newy, _ = name.split('_')
                    index = int(index)
                    newx = int(newx.split('x')[-1])
                    newy = int(newy.split('y')[-1])
                    shape = MAP.shape

                    line = [name, count, index, newx, newy, shape]
                    # print(line)

                    coor_list.append(line)
                    df = pd.DataFrame(coor_list)

                    np.save(os.path.join(MAP_save_path, name), MAP)
                    MAP[MAP > 0] = 255
                    cv2.imwrite(os.path.join(MAP_save_path, name+'.png'), MAP)

                    np.save(os.path.join(VALUE_save_path, name), VALUE)
                    plt.imshow(VALUE), plt.axis('off')
                    plt.savefig(os.path.join(VALUE_save_path, name))
                    # cv2.imwrite(os.path.join(VALUE_save_path, name+'.png'), VALUE)
                break

    df.columns = ['name', 'count', 'index', 'newx', 'newy', 'shape']
    coor_save_name = 'CDcomb.csv'
    df.to_csv(os.path.join(coor_save_path, coor_save_name))
    print(df)


if __name__ == '__main__':
    P6(root='1406542')
