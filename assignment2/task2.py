from pathlib import Path
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
import skimage.io as io
from skimage.filters import gaussian
from scipy.ndimage import median_filter
from scipy.signal import convolve2d
from scipy.interpolate import interp2d
from pdb import set_trace
from itertools import product
from matplotlib import image

from bilateral import bilateral2d
from fspecial import fspecial_gaussian_2d
from nlm import nonlocalmeans

final = io.imread('lighthouse.png').astype(float)/255
raw_noisy = io.imread('lighthouse_RAW_noisy_sigma0.01.png').astype(float)/255


def demosaic(img):
    H, W = img.shape[0], img.shape[1]
    R = np.zeros((H, W))
    G = np.zeros((H, W))
    B = np.zeros((H, W))

    Rx = np.array(range(0, H, 2)) #np.linspace(0, H - 1, H/2)
    Ry = np.array(range(0, W, 2)) #np.linspace(0, W - 1, W/2)
    Bx = np.array(range(1, H, 2)) #np.linspace(1, H - 1, )
    By = np.array(range(1, W, 2))

    Rxn, Ryn = np.meshgrid(Rx, Ry)
    Bxn, Byn = np.meshgrid(Bx, By)

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
    print(R.shape, B.shape, G.shape)
    
    dmosai = (np.stack((R, G, B), axis = 2))#.astype(int)
    # print(dmosai)
    image.imsave('test.png', dmosai)

demosaic(raw_noisy)
