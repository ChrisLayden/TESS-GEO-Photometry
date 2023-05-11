# Chris Layden

import numpy as np
import matplotlib.pyplot as plt
import pysynphot as S
from Observatory import Sensor, Telescope, Observatory
from Instruments import imx455, imx487, mono_tele_v10, mono_tele_v11
from scipy import special, integrate
from PSFs import *

if __name__ == '__main__':
    mag_sp = S.FlatSpectrum(fluxdensity=24.0, fluxunits='abmag')
    mag_sp.convert('fnu')
    gauss_psf_sigma = 1.848
    v10_bandpass = S.UniformTransmission(0.693)
    mono_tele_test = Telescope(diam=25, f_num=8, bandpass=v10_bandpass)
    tess_geo_obs = Observatory(telescope=mono_tele_test, sensor=imx455,
                               filter_bandpass=1., exposure_time=120,
                               num_exposures=1,
                            #    psf_sigma=gauss_psf_sigma)
                               psf_sigma=None)

    print(tess_geo_obs.snr(mag_sp))
    
    # x_start = (np.random.uniform() - 1/2) * 3.76
    # y_start = (np.random.uniform() - 1/2) * 3.76
    # print(x_start, y_start)
    x_start = 0
    y_start = 0
    base_grid = tess_geo_obs.intensity_grid(mag_sp, pos=[x_start,y_start])
    pix_grid = base_grid.reshape((11, 101, 11, 101)).sum(axis=(1, 3))
    noise_per_pix = tess_geo_obs.single_pix_noise()
    aper = optimal_aperture(pix_grid, noise_per_pix)
    print(aper)
    
    # fig, ax = plt.subplots(1)
    # fig.set_figwidth(8)
    # ax.imshow(np.log(pix_grid))
    # for (j,i),label in np.ndenumerate(pix_grid):
    #     lab = format(label / 10 ** 6, '3.2f')
    #     ax.text(i,j,lab,ha='center',va='center')
    # plt.show()
    
    n_aper = aper.sum()

    # The RMS jitter on the timescale of one exposure, in units of pixels
    jitter = 0.5

    num_obs = 900
    sigs = np.zeros(num_obs)
    for i in range(num_obs):
        angle = np.random.uniform(low=0.0, high=2*np.pi)
        displacement = np.random.normal(scale=jitter * 101)
        x_shift = round(np.cos(angle) * displacement)
        y_shift = round(np.sin(angle) * displacement)
        shifted_grid = np.roll(np.roll(base_grid, x_shift, axis=1),
                               -y_shift, axis=0)
        sig = tess_geo_obs.obs_signal(aper, shifted_grid)
        sigs[i] = sig
    
    avg_sig = np.mean(sigs)
    expected_noise = np.sqrt(avg_sig + n_aper * noise_per_pix)
    actual_noise = np.std(sigs)
    phot_prec = actual_noise / avg_sig * 10 ** 6 / np.sqrt(30)
    non_jitter_prec = expected_noise / avg_sig * 10 ** 6 / np.sqrt(30)
    print("Mean Signal (e-): ", format(avg_sig, '4.1e'))
    print("Photometric Precision (ppm): ", format(phot_prec, '4.1f'))
    print("Expected Non-Jitter Precision (ppm): ", format(non_jitter_prec, '4.1f'))


    # # reshape the array into num_stacks sub-arrays, each with 30 elements
    # sub_arrays = signal_list.reshape((num_stacks, 30))

    # # plt.scatter(range(num_stacks * 30), signal_list)
    # # plt.scatter(range(0, num_stacks * 30, 30), averages)
    # # plt.show()
