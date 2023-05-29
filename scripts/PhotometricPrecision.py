# Chris Layden

import numpy as np
import matplotlib.pyplot as plt
import pysynphot as S
from Observatory import Sensor, Telescope, Observatory
from Instruments import imx455, imx487, mono_tele_v10, mono_tele_v11
from PSFs import *
import time
from multiprocessing import Pool
from functools import partial

def jittered_signal(initial_grid, aperture, exposure_time, jitter_time, jitter, resolution=11):
    sub_size = len(initial_grid) // resolution
    num_steps = exposure_time // jitter_time
    angles = np.random.uniform(low=0.0, high=2*np.pi, size=num_steps)
    displacements = np.random.normal(scale=jitter * resolution, size=num_steps)
    x_shifts = np.rint(np.cos(angles) * displacements).astype(int)
    y_shifts = np.rint(np.sin(angles) * displacements).astype(int)
    final_grid = np.zeros_like(initial_grid)
    N, M = initial_grid.shape
    for i in range(num_steps):
        shifted_grid = np.zeros_like(initial_grid)
        shifted_grid[max(x_shifts[i],0):M+min(x_shifts[i],0),
                     max(y_shifts[i],0):N+min(y_shifts[i],0)] = \
                     initial_grid[-min(x_shifts[i],0):M-max(x_shifts[i],0),
                                  -min(y_shifts[i],0):N-max(y_shifts[i],0)]
        final_grid += shifted_grid
    final_grid /= num_steps
    image = final_grid.reshape((sub_size, resolution, sub_size, resolution)).sum(axis=(1, 3))
    obs_image = image * aperture
    return obs_image.sum()


def jittered_signal_list(num_images, initial_grid, aperture, exposure_time, jitter_time, jitter, resolution=11):
    signal_list = np.zeros(num_images)
    for i in range(num_images):
        sig = jittered_signal(initial_grid, aperture, exposure_time, jitter_time, jitter, resolution=resolution)
        signal_list[i] = sig
    return signal_list

def run_jitter(num_images, initial_grid, aperture, exposure_time, jitter_time, jitter_list, resolution=11):
    processes_pool = Pool(len(jitter_list))
    new_func = partial(jittered_signal_list, num_images, initial_grid, aperture, exposure_time, jitter_time, resolution=resolution)
    out_list = processes_pool.map(new_func, jitter_list)
    means_list = np.mean(out_list, axis=1)
    stds_list = np.std(out_list, axis=1)
    return means_list, stds_list


if __name__ == '__main__':
    filter_bp = S.UniformTransmission(1.0)
    # filter_bp = S.ObsBandpass('johnson,b')
    tess_geo_obs = Observatory(telescope=mono_tele_v10, sensor=imx455,
                            filter_bandpass=filter_bp, exposure_time=0.1,
                            num_exposures=300, psf_sigma=None)

    def pix2angle(pix):
        return tess_geo_obs.pix_scale * pix

    def angle2pix(angle):
        return angle / tess_geo_obs.pix_scale

    sub_size = 11
    res = 11
    fig, ax = plt.subplots(layout='constrained')
    jitter_list = np.linspace(0.1, 2.0, 10)
    mag_list = [6]
    for mag in mag_list:
        mag_sp = S.FlatSpectrum(fluxdensity=mag, fluxunits='abmag')
        mag_sp.convert('fnu')
        base_grid = tess_geo_obs.intensity_grid(mag_sp, pos=[0,0], subarray_size=sub_size, resolution=res)
        noise_per_pix = tess_geo_obs.single_pix_noise()
        pix_grid = base_grid.reshape((sub_size, res, sub_size, res)).sum(axis=(1, 3))
        aper = optimal_aperture(pix_grid, noise_per_pix)
        print(aper)
        n_aper = aper.sum()
        sig_list, jitter_stds_list = run_jitter(1000, base_grid, aper, 120, 1, jitter_list)
        noise_list = np.sqrt(sig_list + (n_aper * noise_per_pix) ** 2 + jitter_stds_list ** 2)
        ppm_list = noise_list / sig_list * 10**6
        ax.plot(jitter_list, ppm_list, 'o-')
    
    ax.set_title('Photometric Precision for 2-minute Exposures')
    ax.legend(mag_list, title='Source Magnitude')
    ax.set_xlabel(r'1-$\sigma$ Pointing Precision (pix)')
    secax = ax.secondary_xaxis('top', functions=(pix2angle, angle2pix))
    secax.set_xlabel(r'1-$\sigma$ Pointing Precision (arcsec)')
    ax.set_ylabel('Photometric Precision (ppm)')
    ax.set_yscale('log')
    plt.show()

    # print(phot_prec(3600, 5, 1, mag_sp, [0,0], 0.25, tess_geo_obs))
    # print(time.time() - t1)
    # num_images = 10
    # t0 = time.time()
    # signal_list = np.zeros(num_images)
    # for i in range(num_images):
    #     signal_list[i] = jittered_signal(base_grid, aper, 0.1, 120, 1, 1)
    # t1 = time.time()
    # signal_list2 = run_jitter(num_images, base_grid, aper, 0.1, 120, 1)
    # print(time.time() - t1, t1 - t0)
    # print(time.time() - t0)
    # plt.imshow(test)
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
