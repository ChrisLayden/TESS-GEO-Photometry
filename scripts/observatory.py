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
import warnings
import psfs
import numpy as np
import pysynphot as S
from sky_background import bkg_spectrum
from jitter_tools import jittered_array, integrated_stability, get_pointings, shift_values


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
    def __init__(self, diam, f_num, psf_type='airy', spot_size=1.0,
                 bandpass=S.UniformTransmission(1.0)):
        '''Initializing a telescope object.

        Parameters
        ----------
        diam: float
            Diameter of the primary aperture, in cm
        f_num: float
            Ratio of the focal length to diam
        psf_type: string
            The name of the PSF to use. Options are 'airy' and 'gaussian'.
        spot_size: float
            The spot size (i.e., standard distribution of the psf), relative
            to the diffraction limit. Only used for Gaussian PSFs.
        bandpass: pysynphot.bandpass object
            The telescope bandpass as a function of wavelength,
            accounting for throughput and any geometric blocking
            factor

        '''

        self.diam = diam
        self.f_num = f_num
        self.bandpass = bandpass
        self.focal_length = self.diam * self.f_num
        self.psf_type = psf_type
        self.spot_size = spot_size
        self.plate_scale = 206265 / (self.focal_length * 10**4)


class Observatory(object):
    '''Class specifying a complete observatory.'''

    def __init__(
            self, sensor, telescope, filter_bandpass=S.UniformTransmission(1.0),
            exposure_time=1., num_exposures=1, eclip_lat=90,
            limiting_s_n=5., jitter=None, jitter_psd=None):

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
        jitter: float
            The 1-sigma jitter of the telescope, in arcseconds.
            Basically assumes zero power beyond the frequency
            at which jitter is calculated.
            Either this or jitter_psd must be specified, and
            jitter_psd takes precedence.
        jitter_psd: array-like (n x 2)
            The power spectral density of the jitter. Assumes PSD is
            the same for x and y directions, and no roll jitter.
            Contains two columns: the first is the frequency, in Hz, and
            the second is the PSD, in arcseconds^2/Hz.
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

        self.exposure_time = exposure_time
        self.num_exposures = num_exposures
        self.eclip_lat = eclip_lat
        self.limiting_s_n = limiting_s_n
        plate_scale = 206265 / (self.telescope.focal_length * 10**4)
        self.pix_scale = plate_scale * self.sensor.pix_size
        self.jitter = jitter
        self.jitter_psd = jitter_psd
        # Either jitter or jitter_psd must be specified
        if self.jitter is None and self.jitter_psd is None:
            raise ValueError('Must specify either jitter or jitter_psd.')

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
        '''The effective photometric area of the observatory at the pivot wavelength, in cm^2.'''
        tele_area = np.pi * self.telescope.diam ** 2 / 4
        pivot_throughput = np.interp(self.lambda_pivot,
                                     self.bandpass.wave,
                                     self.bandpass.throughput)
        eff_area = tele_area * pivot_throughput
        return eff_area

    def psf_fwhm(self):
        '''The full width at half maximum of the PSF, in microns.'''
        diff_lim_fwhm = 1.025 * self.lambda_pivot * self.telescope.f_num / 10 ** 4
        if self.telescope.psf_type == 'airy':
            fwhm = diff_lim_fwhm
        elif self.telescope.psf_type == 'gaussian':
            fwhm = diff_lim_fwhm * self.telescope.spot_size
        return fwhm

    def central_pix_frac(self):
        '''The fraction of the total signal in the central pixel.'''
        if self.telescope.psf_type == 'gaussian':
            psf_sigma = self.psf_fwhm() / 2.355
            half_width = self.sensor.pix_size / 2
            pix_frac = psfs.gaussian_ensq_energy(half_width, psf_sigma,
                                                 psf_sigma)
        elif self.telescope.psf_type == 'airy':
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
        '''The noise from the background and sensor in one exposure, in e-/pix.'''
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
        bkg_signal = self.bkg_per_pix()
        dark_noise = self.sensor.dark_current * self.exposure_time
        if (bkg_signal + dark_noise) >= self.sensor.full_well:
            raise ValueError('Noise itself saturates detector')

        def saturation_diff(mag):
            '''Difference between the pixel signal and full well capacity.'''
            signal = mag_10_signal * 10 ** ((10 - mag) / 2.5)
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
        if self.telescope.psf_type == 'gaussian':
            psf_sigma = self.psf_fwhm() / 2.355
            cov_mat = [[psf_sigma ** 2, 0], [0, psf_sigma ** 2]]
            psf_grid = psfs.gaussian_psf(img_size, resolution,
                                         self.sensor.pix_size, pos, cov_mat)
        elif self.telescope.psf_type == 'airy':
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

    def observed_frame(self, spectrum, pointings,
                       pos=np.array([0, 0]), img_size=11, resolution=11):
        '''An actual observed frame, with simulated pointing jitter.

        Parameters
        ----------
        spectrum: pysynphot.spectrum object
            The spectrum for which to calculate the intensity grid.
        pointings: array-like
            List of pointing offsets, in subpixels.
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
        avg_grid = jittered_array(initial_grid, pointings)
        frame = avg_grid.reshape((img_size, resolution, img_size,
                                  resolution)).sum(axis=(1, 3))
        return frame

    def observe(self, spectrum, pos=np.array([0, 0]), num_frames=100,
                img_size=11, resolution=11, aper_method='shift_and_add'):
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
            The RMS jitter, in pixels. Assumes fixed RMS.
        num_frames : int (default 100)
            The number of frames to simulate with jitter.
        img_size: int (default 11)
            The extent of the subarray, in pixels.
        resolution: int (default 11)
            The number of sub-pixels per pixel.
        aper_method: string (default 'shift_and_add')
            The method for determining the optimal aperture.
            Options are 'conv_psf' (convolve the PSF with the jitter
            profile to get an optimal aperture) and 'indiv_aper'
            (find the optimal aperture for each individual frame,
            rather than finding one aperture and using it for all
            frames).

        Returns
        -------
        jitter_noise : float
            The noise from jitter, in e-.
        '''
        
        if self.jitter == 0:
            num_frames = 1
        jitter_pix = self.jitter / self.pix_scale if self.jitter is not None else None
        if self.jitter_psd is not None:
            jitter_psd_pix = self.jitter_psd.copy()
            jitter_psd_pix[:, 1] = jitter_psd_pix[:, 1] / self.pix_scale ** 2
        else:
            jitter_psd_pix = None
        pointings_array = get_pointings(self.exposure_time, num_frames, self.exposure_time / 9.0,
                                        img_size, resolution, jitter_psd_pix, jitter_pix)
        signal_list = np.zeros(num_frames)
        
        if aper_method == 'conv_psf':
            initial_grid = self.signal_grid_fine(spectrum, pos,
                                                img_size, resolution)
            if self.jitter_psd is None:
                jitter_sigma = self.jitter / self.pix_scale
            else:
                jitter_time = np.min([self.exposure_time / 9.0, 0.5])
                jitter_freq = 1 / 2 * jitter_time
                jitter_sigma = integrated_stability(jitter_freq, self.jitter_psd[:,0], self.jitter_psd[:,1])
            psf_with_jitter = psfs.jittered_psf(initial_grid, jitter_sigma, resolution=resolution)
            temp_grid = psf_with_jitter.reshape((img_size, resolution, img_size, resolution))
            pixel_grid_jitter = temp_grid.sum(axis=(1, 3))
            aper = psfs.optimal_aperture(pixel_grid_jitter, self.single_pix_noise())
            for i in range(num_frames):
                pointings = pointings_array[i]
                frame = self.observed_frame(spectrum, pointings, pos=pos,
                                            img_size=img_size,
                                            resolution=resolution)
                signal = np.sum(frame * aper)
                signal_list[i] = signal
        elif aper_method == 'shift_and_add':
            if self.jitter_psd is None:
                jitter_sigma = self.jitter / self.pix_scale
            else:
                jitter_time = np.min([self.exposure_time / 9.0, 0.5])
                jitter_freq = 1 / (2 * jitter_time)
                jitter_sigma = integrated_stability(jitter_freq, jitter_psd_pix[:,0], jitter_psd_pix[:,1])
            initial_grid = self.signal_grid_fine(spectrum, pos,
                                                 img_size, resolution)
            image_grid = initial_grid * self.num_exposures
            image_noise = self.single_pix_noise() * np.sqrt(self.num_exposures)
            psf_with_jitter = psfs.jittered_psf(image_grid, jitter_sigma, resolution=resolution)
            temp_grid = psf_with_jitter.reshape((img_size, resolution, img_size, resolution))
            pixel_grid_jitter = temp_grid.sum(axis=(1, 3))
            raw_aper = psfs.optimal_aperture(pixel_grid_jitter, image_noise)
            aper_pads = psfs.get_aper_padding(raw_aper)
            for i in range(num_frames):
                pointings = pointings_array[i]
                del_x = np.mean(pointings[:,0]) / 11
                del_y = np.mean(pointings[:,1]) / 11
                del_x_int = np.rint(np.mean(pointings[:,0]) / 11).astype(int)
                del_y_int = np.rint(np.mean(pointings[:,1]) / 11).astype(int)
                if del_y_int < -aper_pads[0] or del_y_int > aper_pads[1]:
                    raise ValueError('Subarry size too small for jitter.')
                if del_x_int < -aper_pads[2] or del_x_int > aper_pads[3]:
                    raise ValueError('Subarry size too small for jitter.')
                frame = self.observed_frame(spectrum, pointings, pos=pos,
                                            img_size=img_size,
                                            resolution=resolution)
                aper = shift_values(raw_aper, del_x_int, del_y_int)
                signal = np.sum(frame * aper)
                signal_list[i] = signal

        # import matplotlib.pyplot as plt
        # fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
        # fig.set_size_inches(10, 5)
        # ax1.plot(signal_list)
        # # ax2.plot(offset_list)
        # ax2.plot(del_x_list)
        # ax3.plot(del_y_list)
        # # ax2.plot(aper_count_list)
        # plt.show()

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
    telescope_bandpass = S.UniformTransmission(0.758)
    imx455 = Sensor(pix_size=3.76, read_noise=1.65, dark_current=1.5*10**-3,
                    full_well=51000, qe=sensor_bandpass)

    mono_tele_v10uvs = Telescope(diam=25, f_num=4.8, psf_type='airy', bandpass=telescope_bandpass)
    b_bandpass = S.ObsBandpass('johnson,b')
    r_bandpass = S.ObsBandpass('johnson,r')
    vis_bandpass = S.UniformTransmission(1.0)

    flat_spec = S.FlatSpectrum(15, fluxunits='abmag')
    flat_spec.convert('fnu')
    freqs = np.linspace(1 / 60, 5, 10000)
    psd = freqs ** 0 * (1.547 / 2.438) ** 2
    tess_geo_obs = Observatory(telescope=mono_tele_v10uvs, sensor=imx455,
                               filter_bandpass=r_bandpass, eclip_lat=90,
                               exposure_time=1, num_exposures=3600,
                               jitter=1, jitter_psd=np.array([freqs, psd]).T)
