'''Tools for adding jitter to an observation.

Functions
---------
jitter_animation : function
    Animate the jittering of an intensity grid.
avg_intensity_to_frame : function
    Convert an intensity grid to an exposure with random noise.
shift_grid : function
    Shift an intensity grid by a given subpixel displacement.
jittered_array : function
    The intensity grid averaged over steps of jittering.
observed_image : function
    The actual image measured by an observatory.
signal_list : function
    A list of signals observed for a given constant spectrum.
multithread_signal_list : function
    A list of average signals and standard deviations observed
    for a given list of spectra.
'''

import scipy
import os
import numpy as np
import matplotlib.pyplot as plt
import pysynphot as S
import psfs
import time
from tools import FakeLC
# from observatory import Observatory, Sensor, Telescope
# from instruments import sensor_dict, telescope_dict
from multiprocessing import Pool
from functools import partial
from matplotlib.animation import FuncAnimation

sub_size = 11
resolution = 11
jitter_time = 1

def shift_values(arr, del_x, del_y):
    '''Shift values in an array by a specified discrete displacement.
    
    Parameters
    ----------
    arr : array-like
        The intensity grid.
    del_x : int
        The x displacement, in subpixels.
    del_y : int
        The y displacement, in subpixels.
        
    Returns
    -------
    new_arr : array-like
        The shifted array.
    '''
    N, M = arr.shape
    new_arr = np.zeros_like(arr)
    new_arr[max(del_x, 0):M+min(del_x, 0), max(del_y, 0):N+min(del_y, 0)] = \
        arr[-min(del_x, 0):M-max(del_x, 0), -min(del_y, 0):N-max(del_y, 0)]
    return new_arr

def jittered_array(arr, num_steps, jitter):
    '''Jitter the values in an array and take the average array.

    Parameters
    ----------
    arr : array-like
        The initial intensity grid.
    num_steps : int
        The number of steps to average over.
    jitter : float
        The RMS jitter, in pixels.

    Returns
    -------
    avg_arr : array-like
        The final intensity grid.
    '''
    angles = np.random.uniform(low=0.0, high=2*np.pi, size=num_steps)
    displacements = np.random.normal(scale=jitter * resolution, size=num_steps)
    del_x_list = np.rint(np.cos(angles) * displacements).astype(int)
    del_y_list = np.rint(np.sin(angles) * displacements).astype(int)
    avg_arr = np.zeros_like(arr)
    for i in range(num_steps):
        shifted_arr = shift_values(arr, del_x_list[i], del_y_list[i])
        avg_arr += shifted_arr
    avg_arr /= num_steps
    return avg_arr

def jitter_animation(initial_grid, exposure_time, jitter):
    '''Animate the jittering of an intensity grid.

    Parameters
    ----------
    initial_grid : array-like
        The initial intensity grid.
    exposure_time : float
        The total exposure time, in seconds.
    jitter : float
        The RMS jitter, in pixels.

    Returns
    -------
    ani : matplotlib.animation.FuncAnimation
        The animation object.
    '''
    num_steps = exposure_time // jitter_time
    angles = np.random.uniform(low=0.0, high=2*np.pi, size=num_steps)
    displacements = np.random.normal(scale=jitter * resolution,
                                     size=num_steps)
    x_shifts = np.rint(np.cos(angles) * displacements).astype(int)
    y_shifts = np.rint(np.sin(angles) * displacements).astype(int)
    initial_frame = initial_grid.reshape((sub_size, resolution, sub_size,
                                          resolution)).sum(axis=(1, 3))
    N, M = initial_grid.shape

    def update(frame):
        '''Do one step in the animation.'''
        i = frame
        shifted_grid = np.zeros_like(initial_grid)
        shifted_grid[max(x_shifts[i], 0):M+min(x_shifts[i], 0),
                     max(y_shifts[i], 0):N+min(y_shifts[i], 0)] = \
            initial_grid[-min(x_shifts[i], 0):M-max(x_shifts[i], 0),
                         -min(y_shifts[i], 0):N-max(y_shifts[i], 0)]
        return shifted_grid

    def stacked_intensity_grid(end_step):
        grid = np.zeros_like(initial_grid)
        for step in range(end_step):
            grid += update(step)
        return grid / end_step

    def stacked_image(end_step):
        grid = np.zeros_like(initial_grid)
        for step in range(end_step):
            grid += update(step)
        image = grid.reshape((sub_size, resolution, sub_size,
                              resolution)).sum(axis=(1, 3))
        return image / end_step

    anim_fig, (anim_ax1, anim_ax2, anim_ax3) = plt.subplots(1, 3)
    anim_fig.set_size_inches(12, 4)
    anim_ax1.set_title("Intensity Grid in Step")
    anim_ax2.set_title("Total Intensity Grid So Far")
    anim_ax3.set_title("Total Frame So Far")
    ln1, = [anim_ax1.imshow(initial_grid / num_steps)]
    ln2, = [anim_ax2.imshow(initial_grid)]
    ln3, = [anim_ax3.imshow(initial_frame)]
    anim_images = [ln1, ln2, ln3]

    def anim_update(frame):
        shifted_grid = update(frame)
        stacked_grid = stacked_intensity_grid(frame + 1)
        stacked_frame = stacked_image(frame + 1)
        ln1.set_array(shifted_grid / num_steps)
        ln2.set_array(stacked_grid)
        ln3.set_array(stacked_frame)
        return anim_images

    ani = FuncAnimation(anim_fig, anim_update, frames=range(num_steps),
                        blit=True)
    plt.show()
    return ani


def avg_intensity_to_frame(grid):
    '''Convert an intensity grid to an exposure with random noise.

    Parameters
    ----------
    grid : array-like
        The intensity grid of electrons per subpixel.

    Returns
    -------
    frame : array-like
        The exposure of electrons per pixel.
    '''
    frame = grid.reshape((sub_size, resolution, sub_size,
                          resolution)).sum(axis=(1, 3))
    return frame

    # # Add shot noise
    # frame = np.random.poisson(lam=frame)
    # # Add read noise
    # read_noise_array = np.random.normal(loc=0, scale=read_noise,
    #                                     size=(sub_size, sub_size))
    # frame += np.round(read_noise_array).astype(int)
    # # Add background and dark current noise
    # bkg_and_dc_array = np.random.poisson(lam=bkg_plus_dc,
    #                                      size=(sub_size, sub_size))
    # frame += np.round(bkg_and_dc_array).astype(int)
    # # Subtract out the mean background and dark current levels
    # frame -= np.round(bkg_plus_dc * np.ones((sub_size, sub_size))).astype(int)
    # return frame

# def observed_image(observatory, initial_grid, jitter):
#     '''The actual image observed.

#     Parameters
#     ----------
#     observatory : Observatory
#         The observatory object.
#     initial_grid : array-like
#         The initial intensity grid.
#     jitter : float
#         The RMS jitter, in pixels.

#     Returns
#     -------
#     image : array-like
#         The observed image.
#     '''
#     image = np.zeros((sub_size, sub_size))
#     # bkg_plus_dc = observatory.bkg_noise + observatory.dark_noise
#     # read_noise = observatory.sensor.read_noise
#     # When jitter frequency is higher than sampling frequency,
#     # we must shift the PSF at least once within each frame.
#     if jitter_time <= observatory.exposure_time:
#         num_steps = round(observatory.exposure_time // jitter_time)
#         for i in range(observatory.num_exposures):
#             intensity_grid = jittered_array(initial_grid, num_steps, jitter)
#             frame = avg_intensity_to_frame(intensity_grid)
#             image += frame
#     # Otherwise, we approximate the jitter as happening every
#     # few exposures.
#     elif jitter_time > observatory.exposure_time:
#         tot_time = (observatory.exposure_time * observatory.num_exposures)
#         num_jitters = int((tot_time // jitter_time))
#         angles = np.random.uniform(low=0.0, high=2*np.pi, size=num_jitters)
#         displacements = np.random.normal(scale=jitter * resolution,
#                                          size=num_jitters)
#         del_x_list = np.rint(np.cos(angles) * displacements).astype(int)
#         del_y_list = np.rint(np.sin(angles) * displacements).astype(int)
#         intensity_grid = initial_grid
#         jitter_count = 0
#         for frame_count in range(observatory.num_exposures):
#             frame_time = frame_count * observatory.exposure_time
#             if frame_time > jitter_count * jitter_time:
#                 del_x = del_x_list[jitter_count]
#                 del_y = del_y_list[jitter_count]
#                 intensity_grid = shift_grid(initial_grid, del_x, del_y)
#                 jitter_count += 1
#             frame = avg_intensity_to_frame(intensity_grid)
#             image += frame

#     return image


def signal_list(observatory, jitter, spectrum, num_images=1000):
    '''A list of signals observed for a given constant spectrum.

    Parameters
    ----------
    observatory : Observatory
        The observatory object.
    num_images : int
        The total number of images to observe.
    jitter : float
        The RMS jitter, in pixels.
    spectrum : pysynphot.spectrum.SourceSpectrum
        The spectrum of the source.

    Returns
    -------
    sig_list : array-like
        The list of signals observed.
    '''
    initial_grid = observatory.signal_subgrid(spectrum, pos=[0, 0],
                                                  subarray_size=sub_size)
    sig_list = np.zeros(num_images)
    for i in range(num_images):
        image = observed_image(observatory, initial_grid, jitter)
        if i == 0:
            # Calculate the optimal aperture from the first image.
            aper = psfs.optimal_aperture(image, observatory.single_pix_noise())
        signal = np.sum(image * aper)
        sig_list[i] = signal
    return sig_list


def multithread_signal_list(observatory, jitter, spectrum_list):
    '''A list of signals observed for a given list of spectra.

    Parameters
    ----------
    observatory : Observatory
        The observatory object.
    obs_duration : float
        The total observation duration, in seconds.
    jitter : float
        The RMS jitter, in pixels.

    Returns
    -------
    means_list : array-like
        The list of mean signals observed for each spectrum.
    stds_list : array-like
        The list of standard deviations of the signals observed
        for each spectrum.'''
    processes_pool = Pool(len(spectrum_list))
    new_func = partial(signal_list, observatory, jitter)
    out_list = processes_pool.map(new_func, spectrum_list)
    means_list = np.mean(out_list, axis=1)
    stds_list = np.std(out_list, axis=1)
    return means_list, stds_list

# my_lc = FakeLC(2, 100.0, 1.)
# x = np.r_[0:1000:5000j]
# y = my_lc(x)
# plt.plot(x,y,'k.-')

def jittered_snr(observatory, jitter, spectrum, num_images=1000):
    '''The SNR of a given spectrum, with a given jitter.
    
    Parameters
    ----------
    observatory : Observatory
        The observatory object.
    jitter : float
        The RMS jitter, in pixels.
    spectrum : pysynphot.spectrum.SourceSpectrum
        The spectrum of the source.
    num_images : int
        The total number of images to observe.
    
    Returns
    -------
    snr : float
        The SNR of the spectrum.
    '''
    static_noise = observatory.observe(spectrum)[1]
    jittered_signals = signal_list(observatory, jitter, spectrum, num_images)
    tot_noise = np.sqrt(static_noise ** 2 + np.std(jittered_signals) ** 2)
    snr = np.mean(jittered_signals) / tot_noise
    return snr


if __name__ == '__main__':
    data_folder = os.path.dirname(__file__) + '/../data/'
    filter_bp = S.UniformTransmission(1.0)
    # mono_tele_v10 = telescope_dict['Mono Tele V10UVS (Visible Coatings)']
    # imx455 = sensor_dict['IMX 455 (Visible)']
    test_tele = Telescope(diam=50, f_num=8)
    test_qe = S.FileBandpass(data_folder + 'imx455.fits')
    test_sensor = Sensor(pix_size=1, read_noise=0, dark_current=0, qe=test_qe)
    tess_geo_obs = Observatory(telescope=test_tele, sensor=test_sensor,
                               filter_bandpass=filter_bp, exposure_time=60,
                               num_exposures=1, psf_sigma=None)

    mag = 10
    mag_sp = S.FlatSpectrum(fluxdensity=mag, fluxunits='abmag')
    mag_sp.convert('fnu')
    (signal, noise) = tess_geo_obs.observe(mag_sp)[0:2]

    jitter = 0.1

    signals = signal_list(tess_geo_obs, jitter, mag_sp, num_images=1000)
    # test_list = np.zeros(10)
    # for i in range(10):
    #     test_list[i] = np.std(signals[100*i:100*(i+1)])
    # print("Standard deviation of sigma: ", format(np.std(test_list) / np.sqrt(10), ".2f"))
    # plt.hist(signals, bins=50)
    # plt.show()
    print(np.std(signals))
    # tot_noise = np.sqrt(noise ** 2 + np.std(signals) ** 2)
    # print(tot_noise / signal * 10 ** 6)
    print(jittered_snr(tess_geo_obs, 0.1, mag_sp))