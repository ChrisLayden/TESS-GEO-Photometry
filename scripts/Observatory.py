'''Classes and functions for synthetic photometry and noise characterization.

Classes
-------
Sensor
    Class specifying a photon-counting sensor.
Telescope
    Class specifying a telescope.
Observatory
    Class specifying a complete observatory.

Functions
---------
blackbody_spec
    Returns a blackbody spectrum with the desired properties.
'''

import os
import numpy as np
import pysynphot as S
import psfs
import warnings
from sky_background import bkg_spectrum
from jitter_tools import jittered_array


class Sensor(object):
    '''Class specifying a photon-counting sensor.

    Attributes
    ----------
    pix_size: float
        Width of sensor pixels (assumed square), in um
    read_noise: float
        Read noise per pixel, in e-/pix
    dark_current: float
        Dark current at the sensor operating temperature,
        in e-/pix/s
    qe: pysynphot.bandpass object
        The sensor quantum efficiency as a function of wavelength
    full_well: int
        The full well (in e-) of each sensor pixel.
    '''

    def __init__(self, pix_size, read_noise, dark_current,
                 qe=S.UniformTransmission(1), full_well=100000):
        '''Initialize a Sensor object.

        Parameters
        ----------
        pix_size: float
            Width of sensor pixels (assumed square), in um
        read_noise: float
            Read noise per pixel, in e-/pix
        dark_current: float
            Dark current at -25 degC, in e-/pix/s
        qe: pysynphot.bandpass object
            The sensor quantum efficiency as a function of wavelength
        full_well: int
            The full well (in e-) of each sensor pixel.
        '''

        self.pix_size = pix_size
        self.read_noise = read_noise
        self.dark_current = dark_current
        self.full_well = full_well
        self.qe = qe


class Telescope(object):
    '''Class specifying a telescope.

    Attributes
    ----------
    diam: float
        Diameter of the primary aperture, in cm
    f_num: float
        Ratio of the focal length to diam
    bandpass: pysynphot.bandpass object
        The telescope bandpass as a function of wavelength,
        accounting for throughput and any geometric blocking
        factor
    focal_length: float
        The focal length of the telescope, in cm
    plate_scale: float
        The focal plate scale, in um/arcsec
    '''
    def __init__(self, diam, f_num, bandpass=S.UniformTransmission(1)):
        '''Initializing a telescope object.

        Parameters
        ----------
        diam: float
            Diameter of the primary aperture, in cm
        f_num: float
            Ratio of the focal length to diam
        bandpass: pysynphot.bandpass object
            The telescope bandpass as a function of wavelength,
            accounting for throughput and any geometric blocking
            factor

        '''

        self.diam = diam
        self.f_num = f_num
        self.bandpass = bandpass
        self.focal_length = self.diam * self.f_num
        self.plate_scale = 206265 / (self.focal_length * 10**4)


class Observatory(object):
    '''Class specifying a complete observatory.'''

    def __init__(
            self, sensor, telescope, filter_bandpass=1,
            exposure_time=1., num_exposures=1, eclip_lat=90,
            limiting_s_n=5., psf_sigma=None, jitter=0):

        '''Initialize Observatory class attributes.

        Parameters
        ----------
        sensor: Sensor object
            The photon-counting sensor used for the observations.
        telescope: Telescope object
            The telescope used for the observations.
        filter_bandpass: pysynphot.bandpass object
            The filter bandpass as a function of wavelength.
        exposure_time: float
            The duration of each exposure, in seconds.
        num_exposures: int
            The number of exposures to stack into one image.
        eclip_lat: float
            The ecliptic latitude of the target, in degrees.
        limiting_s_n: float
            The signal-to-noise ratio constituting a detection.
        psf_sigma: float
            The standard deviation of the PSF, in microns.
        jitter: float
            The 1-sigma jitter of the telescope, in arcseconds.
            Assumes Gaussian white noise.
        '''

        self.sensor = sensor
        self.telescope = telescope
        self.filter_bandpass = filter_bandpass
        self.bandpass = (filter_bandpass * self.telescope.bandpass *
                         self.sensor.qe)
        # Avoid error when all throughputs are flat
        all_flat_q = (np.all((self.filter_bandpass.wave is None)) and
                      np.all((self.sensor.qe.wave is None)) and
                      np.all(self.telescope.bandpass.wave is None))
        if all_flat_q:
            warnings.warn('All bandpasses are flat. Setting pivot ' +
                          'wavelength based on PYSYNPHOT wavelength ' +
                          'limits (50-2600 nm).')
            wavelengths = S.FlatSpectrum(1, fluxunits='flam').wave
            array_bp = S.ArrayBandpass(wavelengths, np.ones(len(wavelengths)))
            self.bandpass = self.bandpass * array_bp
        self.lambda_pivot = self.bandpass.pivot()
        self.psf_sigma = psf_sigma

        self.exposure_time = exposure_time
        self.num_exposures = num_exposures
        self.eclip_lat = eclip_lat
        self.limiting_s_n = limiting_s_n
        plate_scale = 206265 / (self.telescope.focal_length * 10**4)
        self.pix_scale = plate_scale * self.sensor.pix_size
        self.jitter = jitter

    def binset(self, spectrum):
        '''Narrowest binset from telescope, sensor, filter, and spectrum.'''
        binset_list = [self.filter_bandpass.wave, self.sensor.qe.wave,
                       self.telescope.bandpass.wave, spectrum.wave]
        binset_list = [x for x in binset_list if x is not None]
        range_list = [np.ptp(x) for x in binset_list]
        obs_binset = binset_list[np.argmin(range_list)]
        return obs_binset

    def tot_signal(self, spectrum):
        '''The total number of electrons generated in one exposure.

        Parameters
        ----------
        spectrum: pysynphot.spectrum object
            The spectrum for which to calculate the signal.
        '''
        S.setref(area=np.pi * self.telescope.diam ** 2 / 4)
        obs_binset = self.binset(spectrum)
        obs = S.Observation(spectrum, self.bandpass, binset=obs_binset,
                            force='extrap')
        raw_rate = obs.countrate()
        signal = raw_rate * self.exposure_time
        return signal

    def bkg_per_pix(self):
        '''The background noise per pixel, in e-/pix.'''
        bkg_wave, bkg_ilam = bkg_spectrum(self.eclip_lat)
        bkg_flam = bkg_ilam * self.pix_scale ** 2
        bkg_sp = S.ArraySpectrum(bkg_wave, bkg_flam, fluxunits='flam')
        # # What Frank's been using
        # bkg_sp = S.FlatSpectrum(0.3 * 10 ** -18 * self.pix_scale ** 2,
        #                         fluxunits='flam')
        bkg_signal = self.tot_signal(bkg_sp)
        return bkg_signal

    def eff_area(self):
        '''The effective photometric area of the observatory, in cm^2.'''
        tele_area = np.pi * self.telescope.diam ** 2 / 4
        avg_throughput = self.bandpass.throughput.mean()
        eff_area = tele_area * avg_throughput
        return eff_area

    def psf_fwhm(self):
        '''The full width at half maximum of the PSF, in microns.'''
        if self.psf_sigma is None:
            fwhm = 1.025 * self.lambda_pivot * self.telescope.f_num / 10 ** 4
        else:
            fwhm = 2.355 * self.psf_sigma
        return fwhm

    def central_pix_frac(self):
        '''The fraction of the total signal in the central pixel.'''
        if self.psf_sigma is not None:
            half_width = self.sensor.pix_size / 2
            pix_frac = psfs.gaussian_ensq_energy(half_width, self.psf_sigma,
                                                 self.psf_sigma)
        else:
            half_width = (np.pi * self.sensor.pix_size /
                          (2 * self.telescope.f_num *
                           self.lambda_pivot * 10 ** -4))
            pix_frac = psfs.airy_ensq_energy(half_width)
        return pix_frac

    def single_pix_signal(self, spectrum):
        '''The signal within the central pixel of an image.

        Parameters
        ----------
        spectrum: pysynphot.spectrum object
            The spectrum for which to calculate the signal.
        '''
        pix_frac = self.central_pix_frac()
        signal = self.tot_signal(spectrum) * pix_frac
        return signal

    def single_pix_noise(self):
        '''The noise from the background and sensor, in e-/pix.'''
        # Noise from sensor dark current
        dark_current_noise = np.sqrt(self.sensor.dark_current *
                                     self.exposure_time)
        # Shot noise from the background
        bkg_noise = np.sqrt(self.bkg_per_pix())

        # Add noise in quadrature
        noise = np.sqrt(dark_current_noise ** 2 + bkg_noise ** 2 +
                        self.sensor.read_noise ** 2
                        )
        return noise

    def single_pix_snr(self, spectrum):
        '''The SNR for a given source with a single-pixel aperature.

        Parameters
        ----------
        spectrum: pysynphot.spectrum object
            The spectrum for which to calculate the SNR.
        '''

        signal = self.single_pix_signal(spectrum)
        noise = np.sqrt(signal + self.single_pix_noise() ** 2)
        exposure_snr = signal / noise
        stack_snr = exposure_snr * np.sqrt(self.num_exposures)
        return stack_snr

    def limiting_mag(self):
        '''The limiting AB magnitude for the observatory.'''
        # We use an aperture of just 1 pixel, as this is the optimal
        # aperture for very dark objects, especially for an undersampled
        # system.
        mag_10_spectrum = S.FlatSpectrum(10, fluxunits='abmag')
        mag_10_spectrum.convert('fnu')
        mag_10_signal = self.single_pix_signal(mag_10_spectrum)
        pix_noise = self.single_pix_noise()

        def s_n_diff_mag(mag):
            '''The difference between the S/N at mag and the limiting S/N.'''
            signal = mag_10_signal * 10 ** ((10 - mag) / 2.5)
            noise = np.sqrt(signal + pix_noise ** 2)
            snr = signal / noise * np.sqrt(self.num_exposures)
            return snr - self.limiting_s_n

        # Newton-Raphson method for root-finding
        mag_tol, s_n_tol = 0.01, 0.01
        i = 1
        mag = 15
        mag_deriv_step = 0.01
        eps_mag = 1
        eps_s_n = s_n_diff_mag(mag)
        while abs(eps_s_n) > s_n_tol:
            if abs(eps_mag) < mag_tol:
                raise RuntimeError('No convergence to within 0.01 mag.')
            elif i > 20:
                raise RuntimeError('No convergence after 20 iterations.')
            eps_s_n_prime = ((s_n_diff_mag(mag + mag_deriv_step) - eps_s_n) /
                             mag_deriv_step)
            eps_mag = eps_s_n / eps_s_n_prime
            mag -= eps_mag
            eps_s_n = s_n_diff_mag(mag)
            i += 1
        return mag

    def saturating_mag(self):
        '''The saturating AB magnitude for the observatory.'''
        # We consider just the central pixel, which will saturate first.
        mag_10_spectrum = S.FlatSpectrum(10, fluxunits='abmag')
        mag_10_spectrum.convert('fnu')
        mag_10_signal = self.single_pix_signal(mag_10_spectrum)

        def saturation_diff(mag):
            '''Difference between the pixel signal and full well capacity.'''
            signal = mag_10_signal * 10 ** ((10 - mag) / 2.5)
            bkg_signal = self.bkg_per_pix()
            dark_noise = self.sensor.dark_current * self.exposure_time
            return signal + bkg_signal + dark_noise - self.sensor.full_well

        # Newton-Raphson method for root-finding
        mag_tol, sig_tol = 0.01, 10
        i = 1
        mag = 10
        mag_deriv_step = 0.01
        eps_mag = 1
        eps_sig = saturation_diff(mag)
        while abs(eps_sig) > sig_tol:
            if abs(eps_mag) < mag_tol:
                raise RuntimeError('No convergence to within 0.01 mag.')
            elif i > 100:
                raise RuntimeError('No convergence after 100 iterations.')
            eps_sig_prime = ((saturation_diff(mag + mag_deriv_step) - eps_sig)
                             / mag_deriv_step)
            eps_mag = eps_sig / eps_sig_prime
            mag -= eps_mag
            eps_sig = saturation_diff(mag)
            i += 1
        return mag

    def signal_grid_fine(self, spectrum, pos=np.array([0, 0]),
                         img_size=11, resolution=11):
        '''The average signal (in electrons) produced across a pixel subarray.

        Parameters
        ----------
        spectrum: pysynphot.spectrum object
            The spectrum for which to calculate the intensity grid.
        pos: array-like (default np.array([0, 0]))
            The centroid position of the source on the subarray, in
            microns, with [0,0] representing the center of the
            central pixel.
        img_size: int (default 11)
            The extent of the subarray, in pixels.
        resolution: int (default 11)
            The number of sub-pixels per pixel.
        '''
        tot_signal = self.tot_signal(spectrum)
        if self.psf_sigma is not None:
            cov_mat = [[self.psf_sigma ** 2, 0], [0, self.psf_sigma ** 2]]
            psf_grid = psfs.gaussian_psf(img_size, resolution,
                                         self.sensor.pix_size, pos, cov_mat)
        else:
            psf_grid = psfs.airy_disk(img_size, resolution,
                                      self.sensor.pix_size, pos,
                                      self.telescope.f_num,
                                      self.lambda_pivot)
        intensity_grid = psf_grid * tot_signal
        return intensity_grid

    def signal_grid(self, spectrum, pos=np.array([0, 0]),
                    img_size=11, resolution=11):
        '''The signal (in electrons) in each pixel for a small sensor region.

        Parameters
        ----------
        spectrum: pysynphot.spectrum object
            The spectrum for which to calculate the intensity grid.
        pos: array-like (default [0, 0])
            The centroid position of the source on the subarray, in
            microns, with [0,0] representing the center of the
            central pixel.
        img_size: int (default 11)
            The extent of the subarray, in pixels.
        resolution: int (default 11)
            The number of sub-pixels per pixel.
        '''
        base_grid = self.signal_grid_fine(spectrum, pos, img_size,
                                          resolution)
        temp_grid = base_grid.reshape((11, resolution, 11, resolution))
        pixel_grid = temp_grid.sum(axis=(1, 3))
        return pixel_grid

    def observed_frame(self, spectrum, pos=[0, 0], jitter_time=1,
                       img_size=11, resolution=11):
        '''An actual observed frame, with simulated pointing jitter.

        Parameters
        ----------
        spectrum: pysynphot.spectrum object
            The spectrum for which to calculate the intensity grid.
        pos: array-like (default [0, 0])
            The centroid position of the source on the subarray, in
            microns, with [0,0] representing the center of the
            central pixel.
        img_size: int (default 11)
            The extent of the subarray, in pixels.
        resolution: int (default 11)
            The number of sub-pixels per pixel.

        Returns
        -------
        image : array-like
            The observed image.
        '''

        initial_grid = self.signal_grid_fine(spectrum, pos,
                                             img_size, resolution)
        # Check that jitter sampling frequency is higher than frame rate
        if jitter_time >= self.exposure_time:
            raise ValueError('Jitter sampling frequency must' +
                             'be higher than frame rate')
        num_steps = round(self.exposure_time // jitter_time)
        avg_grid = jittered_array(initial_grid, num_steps, self.jitter)
        frame = avg_grid.reshape((img_size, resolution, img_size,
                                  resolution)).sum(axis=(1, 3))
        return frame

    def observe(self, spectrum, pos=[0, 0], num_images=100,
                     jitter_time=1, img_size=11, resolution=11):
        ''' Monte Carlo estimate of the signal and noise for a spectrum.

        Parameters
        ----------
        spectrum: pysynphot.spectrum object
            The spectrum for which to calculate the intensity grid.
        pos: array-like (default [0, 0])
            The centroid position of the source on the subarray, in
            microns, with [0,0] representing the center of the
            central pixel.
        jitter : float
            The RMS jitter, in pixels. Assumes Gaussian white noise.
        num_images : int (default 1000)
            The number of images to simulate.
        jitter_time : float (default 1)
            The time between jitter samples, in seconds.
        img_size: int (default 11)
            The extent of the subarray, in pixels.
        resolution: int (default 11)
            The number of sub-pixels per pixel.

        Returns
        -------
        jitter_noise : float
            The noise from jitter, in e-.
        '''
        if self.jitter == 0:
            num_images = 1
        signal_list = np.zeros(num_images)
        for i in range(num_images):
            frame = self.observed_frame(spectrum, pos=pos,
                                        jitter_time=jitter_time,
                                        img_size=img_size,
                                        resolution=resolution)
            if i == 0:
                # Calculate the optimal aperture from the first image.
                aper = psfs.optimal_aperture(frame, self.single_pix_noise())
            signal = np.sum(frame * aper)
            signal_list[i] = signal
        signal = np.mean(signal_list) * self.num_exposures
        jitter_noise = np.std(signal_list) * np.sqrt(self.num_exposures)
        shot_noise = np.sqrt(signal)
        n_aper = aper.sum()
        dark_noise = np.sqrt(n_aper * self.num_exposures *
                             self.sensor.dark_current * self.exposure_time)
        bkg_noise = np.sqrt(n_aper * self.num_exposures * self.bkg_per_pix())
        read_noise = np.sqrt(n_aper * self.num_exposures *
                             self.sensor.read_noise ** 2)
        tot_noise = np.sqrt(jitter_noise ** 2 + shot_noise ** 2 +
                            dark_noise ** 2 + bkg_noise ** 2 +
                            read_noise ** 2)
        results_dict = {'signal': signal, 'tot_noise': tot_noise,
                        'jitter_noise': jitter_noise,
                        'dark_noise': dark_noise, 'bkg_noise': bkg_noise,
                        'read_noise': read_noise, 'shot_noise': shot_noise,
                        'n_aper': int(n_aper), 'snr': signal / tot_noise}
        return results_dict


# Some tests of Observatory behavior
if __name__ == '__main__':
    data_folder = os.path.dirname(__file__) + '/../data/'
    sensor_bandpass = S.FileBandpass(data_folder + 'imx455.fits')
    telescope_bandpass = S.UniformTransmission(0.693)
    imx455 = Sensor(pix_size=3.76, read_noise=1.65, dark_current=1.5*10**-3,
                    full_well=51000, qe=sensor_bandpass)

    mono_tele_v10 = Telescope(diam=25, f_num=8, bandpass=telescope_bandpass)
    b_bandpass = S.ObsBandpass('johnson,b')
    v_bandpass = S.ObsBandpass('johnson,v')
    vis_bandpass = S.UniformTransmission(1.0)

    flat_spec = S.FlatSpectrum(15, fluxunits='abmag')
    flat_spec.convert('fnu')

    tess_geo_obs = Observatory(telescope=mono_tele_v10, sensor=imx455,
                               filter_bandpass=vis_bandpass, eclip_lat=90,
                               exposure_time=300, num_exposures=3,
                               jitter=0.0)
    print(tess_geo_obs.observe(flat_spec, [0, 0]))
