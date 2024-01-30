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

from multiprocessing import Pool
from functools import partial
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.fft import fft, ifft, fftfreq
from scipy.signal import periodogram
import os

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
    new_arr[max(del_y, 0):M+min(del_y, 0), max(del_x, 0):N+min(del_x, 0)] = \
        arr[-min(del_y, 0):M-max(del_y, 0), -min(del_x, 0):N-max(del_x, 0)]
    return new_arr

def get_pointings(exposure_time, num_frames, jitter_time, 
                  img_size, resolution, psd=None, pix_jitter=None):
    '''Jitter the values in an array and take the average array.

    Parameters
    ----------
    exposure_time : float
        The duration of each frame, in seconds.
    num_exposures : int
        The number of frames for which to calculate pointings.
    jitter_time : float
        The duration of each jitter step, in seconds.
    img_size : int
        The size of the subgrid, in pixels.
    resolution : int
        The number of subpixels per pixel in the subgrid.
    psd : array-like (optional)
        The power spectral density of the jitter, in pix^2/Hz.
        If not specified, the jitter will be white noise.
    pix_jitter : float (optional)
        The RMS jitter of white noise, in pixels. Must be specified
        if psd is not specified. If both are specified, psd will be
        used.

    Returns
    -------
    pointings : array-like
        An array containing the pointings for each frame.
    '''
    num_steps_frame = int(exposure_time / jitter_time)
    tot_time = exposure_time * num_frames
    tot_steps = num_steps_frame * num_frames
    times = np.linspace(0, tot_time, tot_steps + 1)
    jitter_freq = 1 / (2 * jitter_time)
    if psd is not None:
        freqs = psd[:, 0]
        psd_arr = psd[:, 1]
        one_sigma = integrated_stability(jitter_freq, freqs, psd_arr)
        # The duration between time steps in the time series generated
        # by the PSD
        psd_time_step = 1 / (2 * np.max(freqs))
        # Check 2 things with the PSD:
        # 1. Make sure that the maximum frequency is high enough to
        #    yield fast enough time steps.
        # 2. Make sure that the PSD has high enough resolution to yield
        #    a long enough time series. If resolution is too low, interpolate
        #    the PSD.
        if psd_time_step > jitter_time:
            raise ValueError("PSD maximum frequency too low for jitter time step.")
        psd_length_required = int(tot_time / psd_time_step / 2 + 1)
        if len(freqs) < psd_length_required:
            freqs_new = np.linspace(np.min(freqs), np.max(freqs), psd_length_required)
            psd_arr = np.interp(freqs_new, freqs, psd_arr)
            freqs = freqs_new
        raw_times_x, time_series_x = psd_to_series(freqs, psd_arr)
        raw_times_y, time_series_y = psd_to_series(freqs, psd_arr)
        # Scale the displacement time series to subpixels, and evaluate
        # at times corresponding to the jitter steps.
        del_x_list = np.zeros(tot_steps)
        del_y_list = np.zeros(tot_steps)
        for i in range(tot_steps):
            del_x_list[i] = np.mean(time_series_x[(raw_times_x >= times[i]) & (raw_times_x < times[i+1])])
            del_y_list[i] = np.mean(time_series_y[(raw_times_y >= times[i]) & (raw_times_y < times[i+1])])
        del_x_list = np.rint(del_x_list * resolution).astype(int)
        del_y_list = np.rint(del_y_list * resolution).astype(int)
    elif pix_jitter is not None:
        one_sigma = pix_jitter
        del_x_list = np.rint(np.random.normal(scale=pix_jitter * resolution,
                                              size=tot_steps)).astype(int)
        del_y_list = np.rint(np.random.normal(scale=pix_jitter * resolution,
                                              size=tot_steps)).astype(int)
    else:
        raise ValueError("Must specify either jitter PSD or pix_jitter.")
    if (2.5 * one_sigma) > (img_size / 2):
        raise ValueError("Jitter too large for subarray.")
    pointings = np.array(list(zip(del_x_list, del_y_list)))
    # Reshape the pointings so each row corresponds to a frame
    pointings_array = pointings.reshape((num_frames, num_steps_frame, 2))
    return pointings_array

def jittered_array(arr, pointings):
    '''Jitter the values in an array and take the average array.

    Parameters
    ----------
    arr : array-like
        The initial intensity grid.
    pointings : array-like
        An array containing the pointings for each frame.

    Returns
    -------
    avg_arr : array-like
        The final intensity grid.
    '''
    num_steps = pointings.shape[0]
    avg_arr = np.zeros_like(arr)
    for i in range(num_steps):
        shifted_arr = shift_values(arr, pointings[i, 0], pointings[i, 1])
        avg_arr += shifted_arr
    avg_arr /= num_steps
    return avg_arr

def series_to_psd(time_series, time_step):
    '''Find the power spectral density of a time series.
    
    Parameters
    ----------
    time_series : array-like
        The values at each time step.
    time_step : float
        The time step between samples, in seconds.
    
    Returns
    -------
    freq_arr : array-like
        The array of frequencies.
    psd_arr : array-like
        The array of power spectral densities.
    '''
    f, P = periodogram(time_series, 1 / time_step, 'boxcar', detrend=False)
    # Only return the positive frequencies
    return f[1:], P[1:]

# Go from a general PSD to a time sequence
def psd_to_series(freq_arr, psd_arr):
    '''Convert a power spectral density to a time series.
    
    Parameters
    ----------
    freq_arr : array-like
        The array of frequencies.
    psd_arr : array-like
        The array of power spectral densities.
        
    Returns
    -------
    times : array-like
        The times, starting at 0 and spaced by 1/(2*max(freq_arr))
    time_series : array-like
        The values at each time step.
    '''
    # First must make sure that frequency values are equally spaced
    freq_arr_spaced = np.linspace(np.min(freq_arr), np.max(freq_arr), len(freq_arr))
    psd_arr_spaced = np.interp(freq_arr_spaced, freq_arr, psd_arr)
    freq_arr = freq_arr_spaced
    psd_arr = psd_arr_spaced
    # Get rid of any DC offset
    if freq_arr[0] == 0:
        freq_arr = freq_arr[1:]
        psd_arr = psd_arr[1:]
    N = len(freq_arr)
    Fs = 2 * np.max(freq_arr)
    time_step = 1 / Fs
    phases = np.random.uniform(low=0.0, high=2*np.pi, size=len(freq_arr))
    amplitude_arr = np.sqrt(psd_arr * N * Fs) * np.exp(1j * phases)
    # Take the double-sided IFFT, where for negative frequencies we take the
    # complex conjugate of the amplitudes for the respective positive frequencies.
    # This eliminates the imaginary part of the time series.
    new_amplitude_arr = np.concatenate(([0], amplitude_arr, np.conjugate(amplitude_arr[::-1])))
    time_series = np.fft.ifft(new_amplitude_arr)
    times = np.linspace(0, 2 * N * time_step, 2 * N + 1)
    time_series = np.real(time_series)
    return times, time_series

def integrated_stability(freq, freq_arr, psd_arr, sigma_level=1):
    '''Find the integrated stability of a PSD at a given frequency.
    
    Parameters
    ----------
    freq_arr : array-like
        The array of frequencies, in Hz.
    psd_arr : array-like
        The array of power spectral densities, in arcsec^2/Hz.
    freq : float
        The frequency at which to calculate the stability, in Hz.
    sigma_level : float (default 1)
        The number of standard deviations to report. Default gives
        RMS stability
        
    Returns
    -------
    stability : float
        The integrated stability, in arcsec.
    '''
    integrated_psd = np.trapz(psd_arr[freq_arr < freq], freq_arr[freq_arr < freq])
    stability = np.sqrt(integrated_psd) * sigma_level
    return stability

# def multithread_signal_list(observatory, jitter, spectrum_list):
#     '''A list of signals observed for a given list of spectra.

#     Parameters
#     ----------
#     observatory : Observatory
#         The observatory object.
#     obs_duration : float
#         The total observation duration, in seconds.
#     jitter : float
#         The RMS jitter, in pixels.

#     Returns
#     -------
#     means_list : array-like
#         The list of mean signals observed for each spectrum.
#     stds_list : array-like
#         The list of standard deviations of the signals observed
#         for each spectrum.'''
#     processes_pool = Pool(len(spectrum_list))
#     new_func = partial(signal_list, observatory, jitter)
#     out_list = processes_pool.map(new_func, spectrum_list)
#     means_list = np.mean(out_list, axis=1)
#     stds_list = np.std(out_list, axis=1)
#     return means_list, stds_list

if __name__ == '__main__':
    # Check that the PSD to time series function works by taking
    # the PSD of the series
    freqs = np.linspace(1/60, 5, 100000)
    psd = freqs ** -1
    import time
    start = time.time()
    raw_times, time_series = psd_to_series(freqs, psd)
    end = time.time()
    print(end - start)
    

    # Load TESS jitter data
    data_folder = os.path.dirname(__file__) + '/../data/'
    tess_data = np.genfromtxt(data_folder + 'TESS_Jitter_PSD.csv', delimiter=',')
    tess_freq = tess_data[:, 0]
    tess_psd = tess_data[:, 1]
    tess_times, tess_time_series = psd_to_series(tess_freq, tess_psd)
    # # Plot the integrated 1-sigma stability vs. frequency
    # tess_stability = np.zeros(len(tess_freq))
    # for i, freq in enumerate(tess_freq):
    #     tess_stability[i] = integrated_stability(freq, tess_freq, tess_psd)
    # plt.plot(tess_freq, tess_stability)
    # plt.xscale('log')
    # plt.yscale('log')
    # plt.show()