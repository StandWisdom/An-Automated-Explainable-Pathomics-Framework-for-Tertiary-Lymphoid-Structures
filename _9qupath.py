import os
import json
import numpy as np
import pandas as pd


def classify(index, name_list, flag, agglist, flilist, fliilist):
    NAME, COLOR = 'None', [0, 0, 0]
    if index in agglist:
        NAME, COLOR = name_list[0], flag[0]
    elif index in flilist:
        NAME, COLOR = name_list[1], flag[1]
    elif index in fliilist:
        NAME, COLOR = name_list[2], flag[2]
    else:
        print('ERROR')
    return {'name': NAME, 'color': COLOR.tolist()}


def give_list(root):
    AGG_path = os.path.join(root, 'p6_2corr_result_smooth/CDcomb.csv')
    FLI_path = os.path.join(root, 'p7coor_result/coor/CD21.csv')
    FLII_path = os.path.join(root, 'p8coor_result/coor/CD23.csv')

    if os.path.exists(AGG_path):
        AGG = pd.read_csv(AGG_path, index_col=0)['name'].values
    else:
        AGG = []
    if os.path.exists(FLI_path):
        FLI = pd.read_csv(FLI_path, index_col=0)['name'].values
    else:
        FLI = []
    if os.path.exists(FLII_path):
        FLII = pd.read_csv(FLII_path, index_col=0)['name'].values
    else:
        FLII = []

    AGG_list, FLI_list, FLII_list = [], [], []

    for flii in FLII:
        inx = int(flii.split('_')[0])
        FLII_list.append(inx)

    for fli in FLI:
        inx = int(fli.split('_')[0])
        if inx not in FLII_list:
            FLI_list.append(inx)

    for agg in AGG:
        inx = int(agg.split('_')[0])
        if (inx not in FLII_list) and (inx not in FLI_list):
            AGG_list.append(inx)

    return AGG_list, FLI_list, FLII_list


def P9_qupath(root):
    print('process9 qupath')

    save_path = os.path.join(root, 'p9_qupath')
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    NAME_LIST = ['Agg', 'FLI', 'FLII']
    FLAG = np.array([
            [255, 255, 255],
            [0, 0, 255],
            [255, 0, 255]
        ])

    coor_root = os.path.join(root, 'p6_3coor_correct')
    paths = os.listdir(coor_root)
    paths.sort()

    AGG_list, FLI_list, FLII_list = give_list(root)

    TLS_num = len(paths)
    record = np.zeros([TLS_num, 2]).astype(int)
    for i in range(TLS_num):
        inx = int(os.path.splitext(paths[i].split('_')[1])[0])
        record[i, 0] = inx
        if inx in AGG_list:
            record[i, 1] = 0
        elif inx in FLI_list:
            record[i, 1] = 1
        elif inx in FLII_list:
            record[i, 1] = 2
    df = pd.DataFrame(record)
    df.columns = ['id', 'type']
    df.to_csv(os.path.join(save_path, 'TLS_type.csv'))

    Dict = []
    for i in range(len(paths)):
        pathi = paths[i]
        indx = int(os.path.splitext(pathi.split('_')[-1])[0])
        coor = np.load(os.path.join(coor_root, pathi))
        coor = coor * 20

        dic = {'type': 'Feature', 'id': str(indx), 'geometry': {}, 'properties': {}}

        properties = dic['properties']
        properties['objectType'] = 'annotation'
        properties['classification'] = {'name': 'None', 'color': [255, 0, 0]}
        cls = classify(index=indx, name_list=NAME_LIST, flag=FLAG, agglist=AGG_list, flilist=FLI_list, fliilist=FLII_list)
        properties['classification'] = cls
        dic['properties'] = properties
        dic['properties']['classification']['name'] = str(indx) + '_' + dic['properties']['classification']['name']

        geometry = dic['geometry']
        geometry['type'] = 'Polygon'
        geometry['coordinates'] = coor.transpose(1, 0, 2).tolist()
        head = geometry['coordinates'][0][0]
        geometry['coordinates'][0].append(head)
        dic['geometry'] = geometry

        Dict.append(dic)

    with open(os.path.join(save_path, root+'.json'), 'w') as f:
        json.dump(Dict, f)
    print('json saved', os.path.join(save_path, root+'.json'))


if __name__ == '__main__':
    root = '1567648'
    P9_qupath(root)

