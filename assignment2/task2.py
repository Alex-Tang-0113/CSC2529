from pathlib import Path
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
import skimage.io as io
from skimage.filters import gaussian
from skimage.color import rgb2ycbcr, ycbcr2rgb
from scipy.ndimage import median_filter
from scipy.signal import convolve2d
from scipy.interpolate import interp2d
from pdb import set_trace
from itertools import product
from matplotlib import image
from scipy.ndimage import median_filter
from bilateral import bilateral2d
from fspecial import fspecial_gaussian_2d
from nlm import nonlocalmeans
from skimage.metrics import peak_signal_noise_ratio as PSNR

original = io.imread('lighthouse.png').astype(float)/255
raw_noisy = io.imread('lighthouse_RAW_noisy_sigma0.01.png').astype(float)/255

# def PSNR(original, restored):
#     mse = np.mean((original - restored) ** 2)
#     if mse == 0:
#         return 100
#     psnr = 10 * np.log10(original.max()**2 / mse)
#     return psnr


def convolve(A, filter, x_range, y_range, img):
    for i in x_range:
        for j in y_range:
            A[i, j] = (img[i-2:i+3, j-2:j+3]*filter).sum()
    
    return A


def demosaic(img):
    H, W = img.shape[0], img.shape[1]
    R = np.zeros((H, W))
    G = np.zeros((H, W))
    B = np.zeros((H, W))

    Gra_R = np.zeros((H, W))
    Gra_B = np.zeros((H, W))
    Gra_G = np.zeros((H, W))

    Rx = np.array(range(0, H, 2)) #np.linspace(0, H - 1, H/2)
    Ry = np.array(range(0, W, 2)) #np.linspace(0, W - 1, W/2)
    Bx = np.array(range(1, H, 2)) #np.linspace(1, H - 1, )
    By = np.array(range(1, W, 2))

    # Rxn, Ryn = np.meshgrid(Rx, Ry)
    # Bxn, Byn = np.meshgrid(Bx, By)

    R[0::2, 0::2] = img[0::2, 0::2]
    B[1::2, 1::2] = img[1::2, 1::2]
    G[1::2, 0::2] = img[1::2, 0::2]
    G[0::2, 1::2] = img[1::2, 0::2]
    #print(R.shape)
    
    xx = np.arange(0, H, 1)
    yy = np.arange(0, W, 1)

    interpolate_R = interp2d(Ry, Rx, R[0::2, 0::2])
    interpolate_B = interp2d(By, Bx, B[1::2, 1::2])
    R = interpolate_R(yy, xx)
    B = interpolate_B(yy, xx)
    
    G_up = np.roll(G, -1, axis = 0)
    G_down = np.roll(G, 1, axis = 0)
    G_left = np.roll(G, -1, axis = 1)
    G_right = np.roll(G, 1, axis = 1)

    G[0::2, 0::2] = ((G_down + G_left + G_up + G_right)/4)[0::2, 0::2]
    G[1::2, 1::2] = ((G_down + G_left + G_up + G_right)/4)[1::2, 1::2]
    # print(R.shape, B.shape, G.shape)
    
    dmosai = (np.stack((R, G, B), axis = 2))#.astype(int)
    # print(dmosai)

    return dmosai

def hq_interpolation(img):

    GaR_filter = np.array([
        [0, 0, -1, 0, 0],
        [0, 0, 2, 0, 0],
        [-1, 2, 4, 2, -1],
        [0, 0, 2, 0, 0],
        [0, 0, -1, 0, 0]
    ])/8

    GaB_filter = GaR_filter

    RaG_RrBc_filter = np.array([
        [0, 0, 1/2, 0, 0],
        [0, -1, 0, -1, 0],
        [-1, 4, 5, 4, -1],
        [0, -1, 0, -1, 0],
        [0, 0, 1/2, 0, 0]
    ])/8

    BaG_BrRc_filter = RaG_RrBc_filter

    RaG_BrRc_filter = np.array([
        [0, 0, -1, 0, 0],
        [0, -1, 4, -1, 0],
        [1/2, 0, 5, 0, 1/2],
        [0, -1, 4, -1, 0],
        [0, 0, -1, 0, 0]
    ])/8

    BaG_RrBc_filter = RaG_BrRc_filter

    RaB_filter = np.array([
        [0, 0, -3/2, 0, 0],
        [0, 2, 0, 2, 0],
        [-3/2, 0, 6, 0, -3/2],
        [0, 2, 0, 2, 0],
        [0, 0, -3/2, 0, 0]
    ])/8

    BaR_filter = RaB_filter


    H, W = img.shape[0], img.shape[1]
    R = np.zeros((H, W))
    G = np.zeros((H, W))
    B = np.zeros((H, W))

    # interpolate
    R[0::2, 0::2] = img[0::2, 0::2]
    B[1::2, 1::2] = img[1::2, 1::2]
    G[1::2, 0::2] = img[1::2, 0::2]
    G[0::2, 1::2] = img[1::2, 0::2]

    # G = convolve(G, GaR_filter, range(2, H-2, 2), range(2, W-2, 2), img)
    # G = convolve(G, GaB_filter, range(3, H-2, 2), range(3, W-2, 2), img)

    G[0::2, 0::2] = convolve2d(img, GaR_filter, boundary='symm',mode='same')[0::2, 0::2]
    G[1::2, 1::2] = convolve2d(img, GaB_filter, boundary='symm',mode='same')[1::2, 1::2]

    # R = convolve(R, RaB_filter, range(3, H-2, 2), range(3, W-2, 2), img)
    # R = convolve(R, RaG_BrRc_filter, range(3, H-2, 2), range(2, W-2, 2), img)
    # R = convolve(R, RaG_RrBc_filter, range(2, H-2, 2), range(3, W-2, 2), img)

    R[1::2, 1::2] = convolve2d(img, RaB_filter, boundary='symm',mode='same')[1::2, 1::2]
    R[1::2, 0::2] = convolve2d(img, RaG_BrRc_filter, boundary='symm',mode='same')[1::2, 0::2]
    R[0::2, 1::2] = convolve2d(img, RaG_RrBc_filter, boundary='symm',mode='same')[0::2, 1::2]
    
    # B = convolve(B, BaR_filter, range(2, H-2, 2), range(2, W-2, 2), img)
    # B = convolve(B, BaG_BrRc_filter, range(3, H-2, 2), range(2, W-2, 2), img)
    # B = convolve(B, BaG_RrBc_filter, range(2, H-2, 2), range(3, W-2, 2), img)

    B[0::2, 0::2] = convolve2d(img, BaR_filter, boundary='symm',mode='same')[0::2, 0::2]
    B[1::2, 0::2] = convolve2d(img, BaG_BrRc_filter, boundary='symm',mode='same')[1::2, 0::2]
    B[0::2, 1::2] = convolve2d(img, BaG_RrBc_filter, boundary='symm',mode='same')[0::2, 1::2]


    dmosai = (np.stack((R, G, B), axis = 2))#.astype(int)

    dmosai[np.where(dmosai > 1)] = 1
    dmosai[np.where(dmosai < 0)] = 0

    # dmosai = (dmosai - min)/(max - min)
    # print(dmosai)

    return dmosai

def gamma_correction(img):
    return img**(1/2.2)

def low_pass_filter(img, size):
    img = (img*255).astype(np.uint8)
    o = rgb2ycbcr(img, channel_axis = -1)
    lum, cb, cr = o[:, :, 0], o[:, :, 1], o[:, :, 2]
    cb = median_filter(cb, size = size)
    cr = median_filter(cr, size = size)

    filtered = ycbcr2rgb(np.stack((lum, cb, cr), axis = 2))

    filtered[np.where(filtered > 1)] = 1
    filtered[np.where(filtered < 0)] = 0
    # max = filtered.max()
    # min = filtered.min()
    # filtered = (filtered - min)/(max - min)
    return filtered



if __name__ == '__main__':
    dmosai = demosaic(raw_noisy)
    q1 = gamma_correction(dmosai)
    print("psnr for q1:", PSNR(original, q1))
    image.imsave('q1.png', q1)

    filtered = low_pass_filter(dmosai, 21)
    q2 = gamma_correction(filtered)
    psnr = PSNR(original, q2)
    print("psnr for q2:", psnr)
    image.imsave('q2.png', q2)

    hq_dmosai = hq_interpolation(raw_noisy)
    q3 = gamma_correction(hq_dmosai)
    print("psnr for q3:", PSNR(original, q3))
    image.imsave('q3.png', q3)

