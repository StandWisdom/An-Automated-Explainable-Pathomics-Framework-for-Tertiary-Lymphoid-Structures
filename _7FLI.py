import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def recgnize(data, CDx):
    data = data-1
    # num = data.max()
    num_list = set(data.reshape(1, -1)[0])
    MAP = np.zeros_like(data)
    VALUE = np.zeros_like(CDx)
    count = 0
    for i in num_list:
        if i <= 0:
            continue
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


def calculate(MAP):
    MAPcopy = np.copy(MAP)
    MAPcopy[MAPcopy > 0] = 255
    MAPcopy = MAPcopy.astype(np.uint8)
    try:
        are_mask, coor = FindMaxRegion(MAPcopy)
    except:
        return 0, 0
    centr_MAP = are_mask * MAP
    centr_MAP_set = set(centr_MAP.reshape([1, -1])[0])
    count2 = len(centr_MAP_set)

    pix_area = (0.1721 * 4) ** 2
    sum_pix = np.sum(centr_MAP > 0)
    sum_area = sum_pix * pix_area
    sum_num = count2
    cell_pix = sum_pix / sum_num
    cell_area = cell_pix * pix_area

    return count2, sum_area


def P7(root, threshold):
    print('process7')
    read_root = os.path.join(root, 'p6_2corr_result_smooth/MAP')  # p6_corr
    paths = os.listdir(read_root)
    paths.sort()
    CDroot = os.path.join(root, 'p3channel/CD21')

    save_root = os.path.join(root, 'p7coor_result')
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
    for i in range(len(paths)):
        pathi = paths[i]
        if '.npy' not in pathi:
            continue
        print(pathi)
        name = os.path.splitext(paths[i])[0]
        mask = np.load(os.path.join(read_root, pathi))
        oriindx = int(pathi.split('_')[0])
        if oriindx >= 50 and oriindx < 100:
            indx = oriindx - 50
        elif oriindx >= 100:
            indx = oriindx - 100
        else:
            indx = oriindx
        CDx = np.load(os.path.join(CDroot, pathi[:3].replace(str(oriindx), str(indx))+pathi[3:]))

        MAP, VALUE, count = recgnize(mask, CDx)
        value_copy = np.copy(VALUE)
        value_copy[value_copy > 0] = 1
        map_copy = np.copy(MAP) * value_copy
        if count != 0:
            lv2_count, lv2_area = calculate(map_copy)
        else:
            lv2_count, lv2_area = 0, 0
        print(lv2_count, lv2_area)

        if lv2_area > threshold:
            coor_list.append(name)
            df = pd.DataFrame(coor_list)

            np.save(os.path.join(MAP_save_path, name), MAP)
            MAP[MAP > 0] = 255
            cv2.imwrite(os.path.join(MAP_save_path, name + '.png'), MAP)

            np.save(os.path.join(VALUE_save_path, name), VALUE)
            plt.imshow(VALUE), plt.axis('off')
            plt.savefig(os.path.join(VALUE_save_path, name))
    try:
        df.columns = ['name']
        coor_save_name = 'CD21.csv'
        df.to_csv(os.path.join(coor_save_path, coor_save_name))
        print(df)
    except:
        return 0


if __name__ == '__main__':
    P7(root='1567648', threshold=500)

'''
patch_size = 500
coor_list = []

num = len(paths)
for i in range(num):
    if '.npy' not in paths[i]:
        continue
    # load DAPI
    img = np.load(os.path.join(root, paths[i]))
    # load CDx
    for CD_path in CD_paths:
        if CD_path == paths[i]:
            print(CD_path)
            CDx = np.load(os.path.join(CD_root, CD_path))
            MAP, VALUE, count = recgnize(img, CDx)
            print(count)
            # plt.figure(0), plt.imshow(MAP)
            # plt.figure(1), plt.imshow(VALUE)
            # plt.show()

            if count > 100:
                name = os.path.splitext(paths[i])[0]
                index, newx, newy, _ = name.split('_')
                index = int(index)
                newx = int(newx.split('x')[-1])
                newy = int(newy.split('y')[-1])

                line = [name, count, index, newx, newy]
                print(line)

                coor_list.append(line)
                df = pd.DataFrame(np.array(coor_list))

                np.save(os.path.join(MAP_save_path, name), MAP)
                MAP[MAP > 0] = 255
                cv2.imwrite(os.path.join(MAP_save_path, name+'.png'), MAP)

                np.save(os.path.join(VALUE_save_path, name), VALUE)
                plt.imshow(VALUE), plt.axis('off')
                plt.savefig(os.path.join(VALUE_save_path, name))
                # cv2.imwrite(os.path.join(VALUE_save_path, name+'.png'), VALUE)

            break

df.columns = ['name', 'count', 'index', 'newx', 'newy']
coor_save_name = 'CD20.csv'
df.to_csv(os.path.join(coor_save_path, coor_save_name))
print(df)
'''
