import os
import warnings
import psfs
import numpy as np
import pysynphot as S
from sky_background import bkg_spectrum
from jitter_tools import jittered_array, integrated_stability, get_pointings, shift_values
import matplotlib.pyplot as plt
import time

# Test Gaussian psf
def test_gaussian_psf():
    '''Test the Gaussian PSF function.'''
    psf = psfs.gaussian_psf(11, 11, 1, np.array([0,0]), np.array([[4,0],[0,4]]))
    missing_frac = 1 - psf.sum()
    diff = abs(0.01188 - missing_frac)
    assert diff < 1e-5, f'Expected different fraction of light in subarray'

def test_optimal_aperture():
    stack_size = 1000
    signal = 100
    img_size = 21
    res = 11
    # Generate a PSF representing the mean value of the signal in each subpixel.
    psf = signal * psfs.gaussian_psf(img_size, res, 1, np.array([0,0]), np.array([[1,0],[0,1]]))
    frame = psf.reshape((img_size, res, img_size, res)).sum(axis=(1, 3))
    read_noise = 0
    dark_noise = 0.01
    bkg_noise = 0
    frame += (dark_noise + bkg_noise)
    frame = np.random.poisson(frame, (stack_size, img_size, img_size))
    read_noise_arr = np.rint(np.random.normal(0, read_noise, (stack_size, img_size, img_size))).astype(int)
    frame += read_noise_arr
    frame = frame.sum(axis=0) - (dark_noise + bkg_noise) * stack_size
    frame[frame < 0] = 0
    # Note: for an array of size ~10000, Poisson sampling is ~0.0005 seconds. Guassian sampling
    # is ~0.0004 seconds.
    noise_per_pix = np.sqrt(read_noise ** 2 + dark_noise + bkg_noise) * np.sqrt(stack_size)
    ap = psfs.get_optimal_aperture(frame, noise_per_pix)
    # signal = (frame * ap).sum()
    # noise = np.sqrt(signal + ap.sum() * (noise_per_pix ** 2))
    plt.imshow(frame)
    plt.colorbar()
    # Put a red x on pixels that are included in the aperture.
    for i in range(img_size):
        for j in range(img_size):
            if ap[i,j] > 0:
                plt.plot(j, i, 'ro')

    plt.show()

test_optimal_aperture()