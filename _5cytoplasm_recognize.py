import os
import cv2
import numpy as np
import matplotlib.pyplot as plt


def EROSION(data, kernel_shape=cv2.MORPH_ELLIPSE, kernel_size=3, iterations=1):
    """
    kernel_shape:
    cv2.MORPH_RECT: 矩形
    cv2.MORPH_CROSS: 十字形(以矩形的锚点为中心的十字架)
    cv2.MORPH_ELLIPSE:椭圆(矩形的内切椭圆）
    """
    kernel = cv2.getStructuringElement(kernel_shape, (kernel_size, kernel_size))
    eros = cv2.erode(data, kernel, iterations=iterations)
    return eros


def recgnize(data, kernel_shape=cv2.MORPH_ELLIPSE, kernel_size=3, iterations=1):
    num = data.max()
    MAP = np.zeros_like(data)
    for i in range(1, num+1):
        indx = np.where(data == i)
        coor = np.vstack([indx[0], indx[1]]).T
        map = np.zeros_like(data)
        for j in range(len(coor)):
            map[coor[j][0], coor[j][1]] = 255
        map = map.astype(np.uint8)
        eromap = EROSION(map, kernel_shape=kernel_shape, kernel_size=kernel_size, iterations=iterations)
        MAP += eromap

    return MAP


def cytoplasm(root, save_path, cell_path, flag, iterations=2):
    filenames = os.listdir(root)
    filenames.sort()
    for filename in filenames:
        if 'ori_' not in filename:
            continue
        else:
            name, _ = os.path.splitext(filename)
            name = name.split('ori_')[1]
            print(name)

            # load data
            # ori_path = os.path.join(root, 'ori_' + name + '.png')
            # ori = cv2.imread(ori_path)[:, :, ::-1]  # RGB
            imgline_path = os.path.join(root, 'imgout_' + name + '.png')
            nuclear = cv2.imread(imgline_path)[:, :, ::-1]  # RGB

            mask_path = os.path.join(root, 'mask_' + name + '.npy')
            mask = np.load(mask_path)  # mask 标记

            # outline_path = os.path.join(root, 'outline_' + name + '.npy')
            # outline = np.load(outline_path)  # 轮廓线

            foreground = recgnize(mask, iterations=iterations).astype(np.uint8)
            if flag:
                plt.figure(1), plt.imshow(foreground)
            '''
            # 二值化
            bina = np.zeros_like(mask)
            bina[mask > 0] = 255
            bina = bina.astype(np.uint8)
            if flag:
                plt.figure(2), plt.imshow(bina)

            # 开运算
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            opening = cv2.morphologyEx(bina, cv2.MORPH_OPEN, kernel, iterations=3)
            if flag:
                plt.figure(3), plt.imshow(bina)

            # 距离变换
            dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
            cv2.normalize(dist_transform, dist_transform, 0, 1.0, cv2.NORM_MINMAX)
            if flag:
                plt.figure(4), plt.imshow(dist_transform)

            # 阈值分割确定前景
            ret, dist_transform_threshold_image = cv2.threshold(dist_transform, 0.4*dist_transform.max(), 255, 0)
            dist_transform_threshold_image = dist_transform_threshold_image.astype(np.uint8)
            if flag:
                plt.figure(5), plt.imshow(dist_transform_threshold_image)

            # 前景腐蚀
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            erosion = cv2.erode(dist_transform_threshold_image, kernel, iterations=1)
            if flag:
                plt.figure(6), plt.imshow(erosion)
            '''

            # 膨胀
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            dilate_image = cv2.dilate(foreground, kernel, iterations=11)
            dilate_image = dilate_image.astype(np.uint8)
            if flag:
                plt.figure(7), plt.imshow(dilate_image)

            # 未确定区域
            unknown = cv2.subtract(dilate_image, foreground)
            if flag:
                plt.figure(8), plt.imshow(unknown)

            # 分水岭标记
            ret2, markers = cv2.connectedComponents(foreground)
            markers = markers + 1
            markers[unknown == 255] = 0
            markers_copy = markers.copy()
            markers_copy[markers == 0] = 150  # 灰色表示未知区域
            markers_copy[markers == 1] = 0  # 黑色表示背景
            markers_copy[markers > 1] = 255  # 白色表示前景
            markers_copy = np.uint8(markers_copy)
            if flag:
                plt.figure(9), plt.imshow(markers)

            # 分水岭处理
            markers = cv2.watershed(cv2.cvtColor(dilate_image, cv2.COLOR_GRAY2BGR), markers)
            cell = np.copy(markers)

            markers[0:1, :] = 1
            markers[-1, :] = 1
            markers[:, 0:1] = 1
            markers[:, -1] = 1
            nuclear[markers == -1] = [0, 0, 255]  # 将边界标记为红色
            if flag:
                plt.figure(10), plt.imshow(nuclear)

            if flag:
                plt.show()

            # input('save pause')

            cv2.imwrite(os.path.join(save_path, name + '.png'), nuclear)
            np.save(os.path.join(cell_path, name + '.npy'), cell)
            cv2.imwrite(os.path.join(cell_path, name + '.png'), cell)
        plt.close()


def P5(root):
    print('process5')
    read_root = os.path.join(root, 'p4_nuclear_result')
    save_path = os.path.join(root, 'p5_cytoplasm_result')
    cell_path = os.path.join(root, 'p5_cell_result')
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if not os.path.exists(cell_path):
        os.makedirs(cell_path)

    flag = False

    cytoplasm(read_root, save_path, cell_path, flag)


if __name__ == '__main__':
    P5(root='1406542')
