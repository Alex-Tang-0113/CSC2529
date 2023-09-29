import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
from pathlib import Path
from numpy.fft import fft2, ifft2, fftshift, ifftshift

hw_dir = Path(__file__).parent

# Load images
img1 = io.imread(hw_dir/'image1.png')
img2 = io.imread(hw_dir/'image2.png')

# Part (a)
W = img1.shape[0]       # = 1001 dots
d = np.array([0.4, 2])  # distances (m)
dpi = 300               # dots per inch

#### YOUR CODE HERE ####
size =  W/300*2.54
ppd = d*100*2*np.tan(0.5/180*np.pi)*dpi/2.54 # pixel per degree

# Part (b)
cpd = 5   # Peak contrast sensitivity location (cycles per degree)
physical_image_freq = cpd/ppd*dpi/25.4 # cycle/mm
image_freq = cpd/ppd # cycle/pixel

halfway_physical_image_freq = physical_image_freq.mean() # cycle/pixel
halfway_image_freq = image_freq.mean() # cycle/mm

print(size, ppd, image_freq, halfway_image_freq) # cycle/pixel

sample_freq = halfway_image_freq 
# sample_freq = image_freq[0]
# sample_freq = image_freq[1]
# Part (c)

def filter(img, sample_freq, mode = "low"):
    imgs = []
    cutoff = int(sample_freq * W)
    centerx, centery = (W/2, W/2)
    # x = np.linspace(-cutoff, cutoff, 2*cutoff + 1)

    for i in range(3):
        freq_img = fftshift(fft2(img[:, :, i], axes=(0, 1)), axes=(0, 1))
        freq_w, freq_h = freq_img.shape
        
        if mode == "low":
            mask = np.zeros((W, W))
            for i in range(W):
                for j in range(W):
                    if (i-centerx)**2 + (j-centery)**2 <= cutoff**2:
                        mask[i, j] = 1
            
            plt.imsave("cutoff.png", (mask*255).astype(int))
        else:
            mask = np.ones((W, W))
            for i in range(W):
                for j in range(W):
                    if (i-centerx)**2 + (j-centery)**2 <= cutoff**2:
                        mask[i, j] = 0

        freq_img *= mask
        new_img = (ifft2(ifftshift(freq_img, axes=(0, 1)), axes=(0, 1)))
        new_img = new_img*255
        new_img = new_img.astype(int)
        imgs.append(new_img)
    
    final = np.stack(imgs, axis= -1)

    return final


#hpf = np.zeros_like(img1)  
#lpf = np.zeros_like(img1)  
img1 = img1.astype(np.float64)/255
img2 = img2.astype(np.float64)/255
# print(img1.shape)

hpf = filter(img2, sample_freq, "high")
lpf = filter(img1, sample_freq, "low")

#### Apply the filters to create the hybrid image
hybrid_img = hpf + lpf  

fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))
axs[0,0].imshow(img2)
axs[0,0].axis('off')
axs[0,1].imshow(hpf, cmap='gray')
axs[0,1].set_title("High-pass filter")
axs[1,0].imshow(img1)
axs[1,0].axis('off')
axs[1,1].imshow(lpf, cmap='gray')
axs[1,1].set_title("Low-pass filter")
plt.savefig("hpf_lpf.png", bbox_inches='tight')
io.imsave("hybrid_image.png", np.clip(hybrid_img, a_min=0, a_max=255.))
