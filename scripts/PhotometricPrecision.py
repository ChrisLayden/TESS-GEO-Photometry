# Chris Layden

import numpy as np
import matplotlib.pyplot as plt
import pysynphot as S
from Observatory import Sensor, Telescope, Observatory
from Instruments import imx455, imx487, mono_tele_v10, mono_tele_v11
from PSFs import *
import os

def jittered_image(full_time, step_time, initial_grid, jitter):
    num_steps = full_time // step_time
    angles = np.random.uniform(low=0.0, high=2*np.pi, size=num_steps)
    displacements = np.random.normal(scale=jitter * 101, size=num_steps)
    x_shifts = np.rint(np.cos(angles) * displacements).astype(int)
    y_shifts = np.rint(np.sin(angles) * displacements).astype(int)
    image_grid = np.zeros_like(initial_grid)
    # grids = np.repeat(initial_grid, [num_steps, num_steps], axis=0)
    # plt.imshow(grids[5])
    # plt.show()
    for i in range(num_steps):
        shifted_grid = np.roll(np.roll(initial_grid, x_shifts[i], axis=1),
                               - y_shifts[i], axis=0)
        image_grid += shifted_grid
    image_grid /= num_steps
    return image_grid

def phot_prec(stack_time, num_stacks, step_time, spectrum, pos, jitter, observatory):
    """The photometric precision, in ppm, of one image."""
    image_time = observatory.exposure_time
    images_in_stack = stack_time // image_time
    total_time = stack_time * num_stacks
    num_images = total_time // image_time
    initial_grid = observatory.intensity_grid(spectrum, pos)
    initial_pix_grid = initial_grid.reshape((11, 101, 11, 101)).sum(axis=(1, 3))
    noise_per_pix = observatory.single_pix_noise()
    aper = optimal_aperture(initial_pix_grid, noise_per_pix)
    signal_list = np.zeros(num_images)
    for i in range(num_images):
        image = jittered_image(image_time, step_time, initial_grid, jitter)
        sig = observatory.obs_signal(aper, image)
        signal_list[i] = sig
    avg_sig = np.mean(signal_list)
    noise = np.std(signal_list)
    phot_prec = noise / avg_sig * 10 ** 6 / np.sqrt(images_in_stack)
    return phot_prec

if __name__ == '__main__':
    mag_sp = S.FlatSpectrum(fluxdensity=15.0, fluxunits='abmag')
    mag_sp.convert('fnu')
    gauss_psf_sigma = 1.848
    tess_geo_obs = Observatory(telescope=mono_tele_v10, sensor=imx455,
                               filter_bandpass=1., exposure_time=120,
                               num_exposures=1,
                            #    psf_sigma=gauss_psf_sigma)
                               psf_sigma=None)

    base_grid = tess_geo_obs.intensity_grid(mag_sp, pos=[0,0])
    # print(phot_prec(3600, 5, 12, mag_sp, [0,0], 0.25, tess_geo_obs))
    # print(jittered_image(120, 12, base_grid, 0.1))
    my_ones = np.ones([3,1111,1111])
    copies = my_ones * base_grid
    test_array = np.array([[[1, 2], [3, 4]],[[5, 6],[7, 8]]])

    
    # plt.imshow(shifted_copes[0])
    # plt.show()
    
    # phot_prec_list = np.zeros(shape=[2,1])
    # jitter_list = np.linspace(0, 1, 2)

    # for k in range(len(jitter_list)):
    #     # The RMS jitter on the timescale of one exposure, in units of pixels
    #     jitter = jitter_list[k]
    #     phot_prec_list[k][0] = jitter

    #     for i in range(len(phot_prec_list[1])-1):
    #         print(k, i)
    #         x_start = (np.random.uniform() - 1/2) * 3.76
    #         y_start = (np.random.uniform() - 1/2) * 3.76
    #         # x_start = 0
    #         # y_start = 0
    #         base_grid = tess_geo_obs.intensity_grid(mag_sp, pos=[x_start,y_start])
    #         pix_grid = base_grid.reshape((11, 101, 11, 101)).sum(axis=(1, 3))
    #         noise_per_pix = tess_geo_obs.single_pix_noise()
    #         aper = optimal_aperture(pix_grid, noise_per_pix)
    #         # print(aper)
            
    #         # fig, ax = plt.subplots(1)
    #         # fig.set_figwidth(8)
    #         # ax.imshow(np.log(pix_grid))
    #         # for (j,i),label in np.ndenumerate(pix_grid):
    #         #     lab = format(label / 10 ** 6, '3.2f')
    #         #     ax.text(i,j,lab,ha='center',va='center')
    #         # plt.show()
            
    #         n_aper = aper.sum()

    #         num_obs = 900
    #         sigs = np.zeros(num_obs)
    #         for j in range(num_obs):
    #             angle = np.random.uniform(low=0.0, high=2*np.pi)
    #             displacement = np.random.normal(scale=jitter * 101)
    #             x_shift = round(np.cos(angle) * displacement)
    #             y_shift = round(np.sin(angle) * displacement)
    #             shifted_grid = np.roll(np.roll(base_grid, x_shift, axis=1),
    #                                 -y_shift, axis=0)
    #             sig = tess_geo_obs.obs_signal(aper, shifted_grid)
    #             sigs[j] = sig
            
    #         avg_sig = np.mean(sigs)
    #         expected_noise = np.sqrt(avg_sig + n_aper * noise_per_pix)
    #         actual_noise = np.std(sigs)
    #         phot_prec = actual_noise / avg_sig * 10 ** 6 / np.sqrt(30)
    #         non_jitter_prec = expected_noise / avg_sig * 10 ** 6 / np.sqrt(30)
    #         phot_prec_list[k][i+1] = phot_prec

    # np.savetxt("phot_prec_list.txt", phot_prec_list)
    # print("Mean Signal (e-): ", format(avg_sig, '4.1e'))
    # print("Photometric Precision (ppm): ", format(phot_prec, '4.1f'))
    # print("Expected Non-Jitter Precision (ppm): ", format(non_jitter_prec, '4.1f'))


    # # reshape the array into num_stacks sub-arrays, each with 30 elements
    # sub_arrays = sigs.reshape((30, 30))
    # averages = sub_arrays.sum(axis=1) / 30
    # print(np.std(averages) / np.mean(averages) * 10**6)

    # # plt.scatter(range(num_stacks * 30), signal_list)
    # # plt.scatter(range(0, num_stacks * 30, 30), averages)
    # # plt.show()
