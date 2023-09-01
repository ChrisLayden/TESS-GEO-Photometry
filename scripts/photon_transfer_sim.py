'''Plot photon transfer curves for a given Observatory.'''

import pysynphot as S
from observatory import Sensor, Telescope, Observatory, blackbody_spec
from instruments import *
import matplotlib.pyplot as plt
import numpy as np


def plot_phot_transfer(obs_dict, spectrum, fixed_time=False):
    '''Plot photon transfer curves for a given Observatory.

    Parameters
    ----------
    obs_list : list of Observatory objects
        The observatories to simulate.
    spectrum : pysynphot.spectrum.SourceSpectrum
        The spectrum to observe.
    fixed_time : bool
        If True, plot the photon transfer curve with variable source intensity
        and fixed exposure time. If False, plot for a fixed source intensity
        and variable exposure time.

    '''
    for key in obs_dict:
        obs = obs_dict[key]
        exposure_time_arr = 10 ** np.linspace(0, 2, 50)
        signal_arr = np.zeros_like(exposure_time_arr)
        noise_arr = np.zeros_like(exposure_time_arr)
        for i, exposure_time in enumerate(exposure_time_arr):
            obs.exposure_time = exposure_time
            (signal, noise, obs_grid) = obs.observation(spectrum)
            if signal > obs.sensor.full_well:
                break
            signal_arr[i] = signal
            noise_arr[i] = noise
        # Drop zeros
        signal_arr = signal_arr[signal_arr != 0]
        noise_arr = noise_arr[noise_arr != 0]
        plt.plot(signal_arr, noise_arr, label=key)
    plt.xlabel('Signal (e$^-$)')
    plt.ylabel('Noise (e$^-$)')
    plt.xscale('log')
    plt.yscale('log')
    plt.legend()
    plt.title('Photon Transfer Curve')
    plt.show()

if __name__ == '__main__':
    vis_obs = Observatory(telescope=mono_tele_v10, sensor=imx455,
                          filter_bandpass=johnson_i)
    swir_obs = Observatory(telescope=mono_tele_v10, sensor=imx990,
                           filter_bandpass=johnson_i)
    swir_obs_low_gain = Observatory(telescope=mono_tele_v10,
                                    sensor=imx990_low_gain,
                                    filter_bandpass=johnson_i)
    spectrum = S.FlatSpectrum(fluxdensity=0.001, fluxunits='Jy')
    obs_dict = {'VIS': vis_obs, 'SWIR': swir_obs, 'SWIR Low Gain': swir_obs_low_gain}
    plot_phot_transfer(obs_dict, spectrum)