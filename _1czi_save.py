import os
import slideio
import numpy as np
import scipy
import cv2
import matplotlib.pyplot as plt


def hist_equalize(img, mx=16400):
    # 计算直方图
    hist, bins = np.histogram(img.flatten(), bins=mx, range=[0, mx])
    peaks, _ = scipy.signal.find_peaks(hist, 10**6)
    results_full = scipy.signal.peak_widths(hist, peaks, rel_height=0.999)
    right_ips = results_full[-1]
    max_ips = right_ips.max()
    print(max_ips)
    hist[:int(max_ips+1)] = 0
    # 计算累积分布函数
    cdf = hist.cumsum()
    # 归一化累积分布函数
    cdf_normalized = (cdf - cdf.min()) * mx / (cdf.max() - cdf.min())
    # cdf_normalized = cdf * hist.max() / cdf.max()
    # 使用累积分布函数进行直方图均衡化
    img[img < int(max_ips+1)] = 0
    equalized_img = np.interp(img.flatten(), bins[:-1], cdf_normalized)
    equalized_img = equalized_img.reshape(img.shape)

    return equalized_img.astype(np.uint16)


def czi_load(path):
    driver = 'CZI'
    slide = slideio.open_slide(path, driver)
    img_name = slide.get_aux_image_names()
    print(img_name)
    print('scene num:', slide.num_scenes)
    sce = slide.get_scene(0)
    print('size:', sce.size, 'channel:', sce.num_channels)
    # for channel in range(sce.num_channels):
    #     print('channel index:', channel, 'Name:', sce.get_channel_name(channel), 'Type:', sce.get_channel_data_type(channel))
    return sce


def save_czi(scene, thum, channel, threshold, name, root):
    rect = scene.rect
    size = (0, int(rect[3]/thum))
    block = scene.read_block(rect, size, channel)
    print(block[:, :, 0].max(), block[:, :, 1].max())
    print(block.shape)

    for t in range(len(channel)):
        print(name[t])
        save_path = os.path.join(root, name[t])
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        iblock = block[:, :, t]
        iblock = hist_equalize(iblock)
        iblock[iblock < threshold] = 0
        np.save(os.path.join(save_path, 'img'), iblock)


def P1(czi_path, thum, threshold):
    print('process1: need czi_path')
    CH_LIST = [1, 2]
    CH_NAME = ['CD20', 'CD3']

    temp = os.path.split(czi_path)[-1]
    temp = temp.replace('_', '-')
    temp = os.path.splitext(temp)[0].split('-')
    for i in temp:
        if (len(i) == 7) and ('AXTC' not in i) and ('czi' not in i):
            root = i
            save_root = i + 'thum'
            break
    save_root = os.path.join(root, save_root)
    print(save_root)

    scene = czi_load(czi_path)
    save_czi(scene, thum, channel=CH_LIST, threshold=threshold, name=CH_NAME, root=save_root)

    return root


if __name__ == '__main__':
    czi_path = r"Z:\胰腺癌\mIHC\1\1721807.czi"
    root = P1(czi_path, thum=20, threshold=16140)
    print(root)
