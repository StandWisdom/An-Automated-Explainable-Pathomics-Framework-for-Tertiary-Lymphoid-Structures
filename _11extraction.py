import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def cell_dens(mask, coor, pm):
    pix_len = pm * 5
    mask_copy = np.copy(mask)
    PIX_AREA = pix_len ** 2
    id_set = set(mask.reshape([1, -1])[0])
    num = len(id_set) - 1
    area_mask = np.zeros_like(mask)
    cv2.drawContours(area_mask, [coor], -1, 1, -1)
    mask_copy[mask_copy > 0] = 1
    pix_num = np.sum(mask_copy)
    area = pix_num * PIX_AREA
    dens = num / area

    return area, dens


def diameter(mask, coor, pm):
    pix_len = pm * 5
    area_mask = np.zeros_like(mask)
    cv2.drawContours(area_mask, [coor], -1, 255, -1)
    area_mask = area_mask.astype(np.uint8)
    contours, _ = cv2.findContours(area_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    ellipse = cv2.fitEllipse(contours[0])
    (x, y), (major_axis, minor_axis), angle = ellipse
    major_axis, minor_axis = major_axis * pix_len, minor_axis * pix_len

    return major_axis, minor_axis


def contour_length(mask, coor, pm):
    pix_len = pm * 5
    area_mask = np.zeros_like(mask)
    cv2.drawContours(area_mask, [coor], -1, 255, -1)
    area_mask = area_mask.astype(np.uint8)
    contours, _ = cv2.findContours(area_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_length = cv2.arcLength(contours[0], closed=True)
    contour_length = contour_length * pix_len

    return contour_length


def maturity(mask_name, root):
    TLS_path = os.path.join(root, 'p9_qupath/TLS_type.csv')
    TLS_map = pd.read_csv(TLS_path, index_col=0)
    indx = int(mask_name.split('_')[0])
    TLt = TLS_map[TLS_map['id'] == indx]['type'].values[0]
    FLI_flag = 1 if TLt == 1 else 0
    FLII_flag = 1 if TLt == 2 else 0

    return TLt, FLI_flag, FLII_flag


def hull_ell(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    combined_contour = np.concatenate(contours)
    convex_hull = cv2.convexHull(combined_contour)
    area_mask = np.zeros_like(mask)
    cv2.drawContours(area_mask, [convex_hull], -1, 255, -1)
    area_mask = area_mask.astype(np.uint8)
    contours, _ = cv2.findContours(area_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    ellipse = cv2.fitEllipse(contours[0])

    return ellipse


def FLII_detail(mask, mask_name, flag, root, pm):
    pix_len = pm * 5
    read_root = os.path.join(root, 'p6_4express_result')
    PIX_AREA = pix_len ** 2
    if flag == 0:
        maxdia, area, ratio = 0, 0, 0
    else:
        mask_copy = np.copy(mask)
        mask_copy[mask_copy > 0] = 1
        mask_name = os.path.splitext(mask_name)[0]
        csv_name = mask_name + '.csv'
        info = pd.read_csv(os.path.join(read_root, csv_name), index_col=0)
        id_list = info[info['CD23'] == 1]['cell_id'].values
        mask_zero = np.zeros_like(mask)
        for id in id_list:
            mask_zero[mask == id] = 1
        mask_zero.astype(np.float32)

        pix_num = np.sum(mask_zero)
        area = pix_num * PIX_AREA
        ratio = pix_num / np.sum(mask_copy)

        mask_zero = (mask_zero * 255).astype(np.uint8)
        ellipse = hull_ell(mask_zero)
        (x, y), (major_axis, minor_axis), angle = ellipse
        major_axis= major_axis * pix_len
        maxdia = major_axis

    return maxdia, area, ratio


def FLI_detail(mask, mask_name, flag, root, pm):
    pix_len = pm * 5
    read_root = os.path.join(root, 'p6_4express_result')
    PIX_AREA = pix_len ** 2
    if flag == 0:
        maxdia, area, ratio = 0, 0, 0
    else:
        mask_copy = np.copy(mask)
        mask_copy[mask_copy > 0] = 1
        mask_name = os.path.splitext(mask_name)[0]
        csv_name = mask_name + '.csv'
        info = pd.read_csv(os.path.join(read_root, csv_name), index_col=0)
        id_list = info[info['CD21'] == 1]['cell_id'].values
        mask_zero = np.zeros_like(mask)
        for id in id_list:
            mask_zero[mask == id] = 1
        mask_zero.astype(np.float32)

        pix_num = np.sum(mask_zero)
        area = pix_num * PIX_AREA
        ratio = pix_num / np.sum(mask_copy)

        mask_zero = (mask_zero * 255).astype(np.uint8)
        ellipse = hull_ell(mask_zero)
        (x, y), (major_axis, minor_axis), angle = ellipse
        major_axis= major_axis * pix_len
        maxdia = major_axis

    return maxdia, area, ratio


def molecule(mask_name, root):
    read_root = os.path.join(root, 'p6_4express_result')
    csv_name = mask_name.replace('.npy', '.csv')
    data = pd.read_csv(os.path.join(read_root, csv_name), index_col=0)
    num_CD20 = np.sum(data['CD20'].values)
    num_CD3 = np.sum(data['CD3'].values)
    num_CD21 = np.sum(data['CD21'].values)
    num_CD23 = np.sum(data['CD23'].values)

    return num_CD20, num_CD3, num_CD21, num_CD23


def P11(root, pm):
    print('process11')
    save_path = os.path.join(root, 'p11_extract')
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    map_root = os.path.join(root, 'p6_2corr_result_smooth/MAP')
    map_paths = os.listdir(map_root)
    map_paths.sort()

    coor_root = os.path.join(root, 'p6_2corr_result_smooth/coor')
    coor_paths = os.listdir(coor_root)
    coor_paths.sort()

    data = []
    for i in range(len(map_paths)):
        mask_name = map_paths[i]
        if '.npy' not in mask_name:
            continue
        indx = mask_name.split('_')[0]
        coor_name = 'coor_' + indx + '.npy'

        mask = np.load(os.path.join(map_root, mask_name))
        coor = np.load(os.path.join(coor_root, coor_name))

        line = []
        line.append(indx)

        area, dens = cell_dens(mask, coor, pm)
        line.append(area)
        line.append(dens)
        # print(area, dens)

        major_axis, minor_axis = diameter(mask, coor, pm)
        line.append(major_axis)
        line.append(minor_axis)
        # print(major_axis, minor_axis)

        lenth = contour_length(mask, coor, pm)
        line.append(lenth)
        # print(lenth)

        TLStype, FLI_flag, FLII_flag = maturity(mask_name, root)
        line.append(TLStype)
        line.append(FLI_flag)
        line.append(FLII_flag)
        # print(TLStype, FLI_flag, FLII_flag)

        maxdia_21, area_21, ratio_21 = FLI_detail(mask, mask_name, FLI_flag, root, pm)
        line.append(maxdia_21)
        line.append(area_21)
        line.append(ratio_21)
        # print(maxdia_21, area_21, ratio_21)

        maxdia_23, area_23, ratio_23 = FLII_detail(mask, mask_name, FLII_flag, root, pm)
        line.append(maxdia_23)
        line.append(area_23)
        line.append(ratio_23)
        # print(maxdia_23, area_23, ratio_23)

        num_CD20, num_CD3, num_CD21, num_CD23 = molecule(mask_name, root)
        if num_CD20 == 0 or num_CD3 == 0:
            ratio_203 = 'NULL'
        else:
            ratio_203 = num_CD20 / num_CD3
        if num_CD23 == 0 or num_CD21 == 0:
            ratio_2321 = 'NULL'
        else:
            ratio_2321 = num_CD23 / num_CD21
        den_CD20, den_CD3 = num_CD20 / area, num_CD3 / area
        den_CD21, den_CD23 = num_CD21 / area, num_CD23 / area
        line.append(num_CD20)
        line.append(num_CD3)
        line.append(ratio_203)
        line.append(den_CD20)
        line.append(den_CD3)
        line.append(num_CD21)
        line.append(num_CD23)
        line.append(ratio_2321)
        line.append(den_CD21)
        line.append(den_CD23)
        # print(num_CD20, num_CD3, num_CD21, num_CD23)

        print(line)
        data.append(line)

        # input('pause')

    df = pd.DataFrame(data)
    df.columns = ['TLS index', 'TLS area(um2)', 'Cell density(num/um2)',
                  'TLS maximum diameter(um)', 'TLS minimum diameter(um)', 'Contour perimeter(um)',
                  'Maturity(Agg:0, FLI:1, FLII:2)', 'FL-I presence(Yes:1, No:0)', 'FL-II presence(Yes:1, No:0)',
                  'FL-I maximum diameter(um)', 'FL-I area(um2)', 'FLI/TLS area ratio',
                  'FL-II maximum diameter(um)', 'FL-II area(um2)', 'FLII/TLS area ratio',
                  'CD20 count(num)', 'CD3 count(num)', 'CD20/CD3',
                  'CD20 density(num/um2)', 'CD3 density(num/um2)',
                  'CD21 count(num)', 'CD23 count(num)', 'CD23/CD21',
                  'CD21 density(num/um2)', 'CD23 density(num/um2)'
                  ]
    df.to_csv(os.path.join(save_path, 'extract_result.csv'))
    print(df)


if __name__ == '__main__':
    P11(root='1461078', pm=0.1721)

