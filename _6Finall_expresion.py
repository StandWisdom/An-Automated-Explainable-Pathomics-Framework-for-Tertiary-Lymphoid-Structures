import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def expresion_rec(data, CDall):
    max_id = data.max()
    num_list = set(data.reshape(1, -1)[0])
    len(num_list)
    # print(num_list)
    # input('pause')
    expre_LIST = []
    for i in num_list:
        if i == 0:
            continue
        # if i % 100 == 0:
        #     print('cell:%d' % i)
        record = np.zeros(5)
        record[0] = i
        indx = (data == i)
        CDx_value = CDall.transpose(1, 2, 0)[indx, :2]
        CD_mean = np.mean(CDx_value, axis=0, keepdims=True)
        flag = np.argmax(CD_mean)
        if flag == 0:
            record[1] = 1
        elif flag == 1:
            record[2] = 1
        if np.sum(CDall[2, indx]) != 0:
            record[3] = 1
        if np.sum(CDall[3, indx]) != 0:
            record[4] = 1
        record = record.astype(int)
        expre_LIST.append(record)

    return expre_LIST


def P6_expression(root):
    print('process6 expression')
    read_root = os.path.join(root, 'p6_2corr_result_smooth/MAP')
    paths = os.listdir(read_root)
    paths.sort()

    CD_root = os.path.join(root, 'p3channel')
    CD_list = ['CD20', 'CD3', 'CD21', 'CD23']

    save_root = os.path.join(root, 'p6_4express_result')
    if not os.path.exists(save_root):
        os.makedirs(save_root)

    value_save_root = os.path.join(root, 'p6_4express_value')
    if not os.path.exists(value_save_root):
        os.makedirs(value_save_root)

    num = len(paths)
    for i in range(num):
        if '.npy' not in paths[i]:
            continue
        name = paths[i]
        print(name)
        # load cell
        data = np.load(os.path.join(read_root, paths[i]))

        # load CD
        CDall = []
        for t in range(len(CD_list)):
            type = CD_list[t]

            oriindx = int(paths[i].split('_')[0])
            if oriindx >= 50 and oriindx < 100:
                indx = oriindx - 50
            elif oriindx >= 100:
                indx = oriindx - 100
            else:
                indx = oriindx

            CD_path = os.path.join(CD_root, type,
                                   paths[i][:3].replace(str(oriindx), str(indx)) + paths[i][3:])
            CD = np.load(CD_path)
            CDall.append(CD)

        CDall = np.array(CDall)

        expr_list = expresion_rec(data, CDall)
        df = pd.DataFrame(expr_list)
        df.columns = ['cell_id', 'CD20', 'CD3', 'CD21', 'CD23']

        expr_save_name = name.replace('.npy', '.csv')
        df.to_csv(os.path.join(save_root, expr_save_name))

        value = np.expand_dims(np.copy(data), -1)
        temp = CDall.transpose(1, 2, 0)
        VALUE = np.dstack([value, temp])
        np.save(os.path.join(value_save_root, name), VALUE)


if __name__ == '__main__':
    P6_expression(root='1567648')

