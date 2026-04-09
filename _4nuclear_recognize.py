import os
import cv2
import numpy as np
from urllib.parse import urlparse
import skimage.io
import matplotlib.pyplot as plt
import matplotlib as mpl
from cellpose.io import logger_setup
from cellpose.utils import masks_to_outlines
from cellpose import models, core, utils, transforms


def mask_overlay(img, masks, colors=None):
    """ overlay masks on image (set image to grayscale)

    Parameters
    ----------------

    img: int or float, 2D or 3D array
        img is of size [Ly x Lx (x nchan)]

    masks: int, 2D array
        masks where 0=NO masks; 1,2,...=mask labels

    colors: int, 2D array (optional, default None)
        size [nmasks x 3], each entry is a color in 0-255 range

    Returns
    ----------------

    RGB: uint8, 3D array
        array of masks overlaid on grayscale image

    """
    if colors is not None:
        if colors.max()>1:
            colors = np.float32(colors)
            colors /= 255
        colors = utils.rgb_to_hsv(colors)
    if img.ndim>2:
        img = img.astype(np.float32).mean(axis=-1)
    else:
        img = img.astype(np.float32)

    HSV = np.zeros((img.shape[0], img.shape[1], 3), np.float32)
    HSV[:,:,2] = np.clip((img / 255. if img.max() > 1 else img) * 1.5, 0, 1)
    hues = np.linspace(0, 1, masks.max()+1)[np.random.permutation(masks.max())]
    for n in range(int(masks.max())):
        ipix = (masks==n+1).nonzero()
        if colors is None:
            HSV[ipix[0],ipix[1],0] = hues[n]
        else:
            HSV[ipix[0],ipix[1],0] = colors[n,0]
        HSV[ipix[0],ipix[1],1] = 1.0
    RGB = (utils.hsv_to_rgb(HSV) * 255).astype(np.uint8)
    return RGB


def image_to_rgb(img0, channels=[0, 0]):
    """ image is 2 x Ly x Lx or Ly x Lx x 2 - change to RGB Ly x Lx x 3 """
    img = img0.copy()
    img = img.astype(np.float32)
    if img.ndim<3:
        img = img[:,:,np.newaxis]
    if img.shape[0]<5:
        img = np.transpose(img, (1,2,0))
    if channels[0]==0:
        img = img.mean(axis=-1)[:,:,np.newaxis]
    for i in range(img.shape[-1]):
        if np.ptp(img[:,:,i])>0:
            img[:,:,i] = np.clip(transforms.normalize99(img[:,:,i]), 0, 1)
            img[:,:,i] = np.clip(img[:,:,i], 0, 1)
    img *= 255
    img = np.uint8(img)
    RGB = np.zeros((img.shape[0], img.shape[1], 3), np.uint8)
    if img.shape[-1]==1:
        RGB = np.tile(img,(1,1,3))
    else:
        RGB[:,:,channels[0]-1] = img[:,:,0]
        if channels[1] > 0:
            RGB[:,:,channels[1]-1] = img[:,:,1]
    return RGB


def nuclear(root, save_path, use_GPU):
    paths = os.listdir(root)
    paths.sort()
    print(paths)
    # input('pause')

    for name in paths:
        if '.npy' in name:
            continue
        print(name)

        img = cv2.imread(os.path.join(root, name))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # plt.imshow(img), plt.axis('off')
        # plt.show()
        # input('pause')
        channels = [0, 0]
        model = models.Cellpose(gpu=use_GPU, model_type='cyto')
        masks, flows, styles, diams = model.eval(img, diameter=None, flow_threshold=None, channels=channels)

        # ori image
        img = image_to_rgb(img, channels=channels)
        # outline
        outlines = masks_to_outlines(masks)
        # imgout
        outX, outY = np.nonzero(outlines)
        imgout = img.copy()
        imgout[outX, outY] = np.array([255, 0, 0])
        # mask
        overlay = mask_overlay(img, masks)

        # save
        print('SAVING...')
        cv2.imwrite(os.path.join(save_path, 'ori_' + name), img[:, :, ::-1])
        cv2.imwrite(os.path.join(save_path, 'imgout_' + name), imgout[:, :, ::-1])
        cv2.imwrite(os.path.join(save_path, 'overlay_' + name), overlay[:, :, ::-1])

        np.save(os.path.join(save_path, 'mask_' + os.path.splitext(name)[0] + '.npy'), masks)
        np.save(os.path.join(save_path, 'outline_' + os.path.splitext(name)[0] + '.npy'), outlines)

        # input('pause')


def P4(root, gpu_id='1'):
    print('process4: select gpu (option)')
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
    use_GPU = core.use_gpu()
    logger_setup()
    mpl.rcParams['figure.dpi'] = 300
    plt.rcParams['figure.figsize'] = (1.5, 1.5)

    dataroot = os.path.join(root, 'p3channel/DAPI')
    save_path = os.path.join(root, 'p4_nuclear_result')
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    nuclear(dataroot, save_path, use_GPU)
    # input('pause')


if __name__ == '__main__':
    P4(root='1459149')
    '''
    # example
    imgs_2D = img
    model = models.Cellpose(gpu=use_GPU, model_type='cyto')
    channels = [0, 0]
    masks, flows, styles, diams = model.eval(imgs_2D, diameter=None, flow_threshold=None, channels=channels)
    
    img = image_to_rgb(img, channels=channels)
    plt.imshow(img), plt.axis('off')
    plt.show()
    
    overlay = mask_overlay(img, masks)
    plt.imshow(overlay), plt.axis('off')
    plt.show()
    
    outlines = masks_to_outlines(masks)
    outX, outY = np.nonzero(outlines)
    imgout = img.copy()
    imgout[outX, outY] = np.array([255, 0, 0])
    plt.imshow(imgout), plt.axis('off')
    plt.show()
    
    # plt.imshow(flows[0]), plt.axis('off')
    # plt.show()
    
    cv2.imwrite('outline.png', imgout[:, :, ::-1])
    cv2.imwrite('ori.png', img[:, :, ::-1])
    cv2.imwrite('mask.png', overlay[:, :, ::-1])
    # cv2.imwrite('cellpose.png', flows[0][:, :, ::-1])
    '''
