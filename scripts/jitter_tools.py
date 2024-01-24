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
    new_arr[max(del_x, 0):M+min(del_x, 0), max(del_y, 0):N+min(del_y, 0)] = \
        arr[-min(del_x, 0):M-max(del_x, 0), -min(del_y, 0):N-max(del_y, 0)]
    return new_arr

def jittered_array(arr, duration, jitter_time, resolution, psd=None, pix_jitter=None):
    '''Jitter the values in an array and take the average array.

    Parameters
    ----------
    arr : array-like
        The initial intensity grid.
    duration : float
        The total duration of the image, in seconds.
    jitter_time : float
        The duration of each jitter step, in seconds.
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
    avg_arr : array-like
        The final intensity grid.
    '''
    num_steps = int(duration / jitter_time)
    times = np.linspace(0, duration, num_steps + 1)
    angles = np.random.uniform(low=0.0, high=2*np.pi, size=num_steps)
    if psd is not None:
        freqs = psd[:, 0]
        psd_arr = psd[:, 1]
        jitter_freq = 1 / (2 * jitter_time)
        one_sigma = integrated_stability(jitter_freq, freqs, psd_arr)
        one_sigma_1Hz = integrated_stability(1, freqs, psd_arr)
        raw_times, time_series = psd_to_series(freqs, psd_arr)
        # Scale the displacement time series to subpixels, and evaluate
        # at times corresponding to the jitter steps.
        displacements = np.zeros(num_steps)
        for i in range(num_steps):
            displacements[i] = np.mean(time_series[(raw_times >= times[i]) & (raw_times < times[i+1])])
        displacements *= resolution
    elif pix_jitter is not None:
        one_sigma = pix_jitter
        one_sigma_1Hz = pix_jitter
        displacements = np.random.normal(scale=pix_jitter * resolution,
                                         size=num_steps)
    else:
        raise ValueError("Must specify either jitter PSD or pix_jitter.")
    # Return error if jitter is too large for the subgrid.
    # Do this if 2.5*sigma is greater than half the subgrid size.
    # This criterion corresponds to 1% of positions being outside the subgrid.
    img_size = float(arr.shape[0])
    if (2.5 * one_sigma * resolution) > (img_size / 2):
        raise ValueError("Jitter too large for subarray.")
    # For any displacement that is too large, set it to the maximum allowed.
    displacements[displacements > (img_size / 2)] = img_size / 2
    del_x_list = np.rint(np.cos(angles) * displacements).astype(int)
    del_y_list = np.rint(np.sin(angles) * displacements).astype(int)
    avg_arr = np.zeros_like(arr)
    for i in range(num_steps):
        shifted_arr = shift_values(arr, del_x_list[i], del_y_list[i])
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
    freqs = np.linspace(1/60, 5, 10000)
    psd = freqs ** -1
    raw_times, time_series = psd_to_series(freqs, psd)
    # times = np.arange(0, 1001, 1)
    # stability = integrated_stability(0.5, freqs, psd)
    # y = np.zeros(len(times) - 1)
    # for i in range(len(times) - 1):
    #     y[i] = np.mean(time_series[(raw_times >= times[i]) & (raw_times < times[i+1])])
    # print(np.std(y), stability)
    # print(np.std(time_series), integrated_stability(5, freqs, psd))
    # freqs2, psd2 = series_to_psd(time_series, raw_times[1] - raw_times[0])
    # plt.scatter(freqs2[1:len(freqs2)//2], psd2[1:len(psd2)//2])
    # plt.scatter(freqs, psd)
    # plt.xscale('log')
    # plt.yscale('log')
    # plt.show()
    # freqs2 = freqs2[1:len(freqs2)//2]
    # psd2 = psd2[1:len(psd2)//2]
    # stability1 = integrated_stability(10, freqs, psd)
    # stability2 = integrated_stability(10, freqs2, psd2)
    # print(stability1, stability2)
    # # Plot the stability vs. frequency
    # stability = np.zeros(len(freqs))
    # for i, freq in enumerate(freqs):
    #     stability[i] = integrated_stability(freq, freqs, psd)
    # plt.plot(freqs, stability)
    # plt.xscale('log')
    # plt.yscale('log')
    # plt.show()

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