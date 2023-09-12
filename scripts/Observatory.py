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
import constants
from redshift_lookup import RedshiftLookup
from sky_background import bkg_spectrum


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
    def __init__(self, diam, f_num, bandpass=1):
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
    '''Class specifying a complete observatory.

    Attributes
    ----------
    sensor: Sensor object
        The sensor used in the observatory.
    telescope: Telescope object
        The telescope used in the observatory.
    bandpass: pysynphot.bandpass object
        The bandpass of the observatory as a whole.
        This is the product of the sensor, telescope, and
        filter bandpasses.
    exposure_time: float
        Duration for one exposure, in seconds.
    num_exposures: int (default 1)
        Number of exposures in a stacked image.
    limiting_s_n: float
        The minimum signal to noise ratio for a detection.
    psf_sigma: float
        The standard deviation of the point spread function
        of the system, in um, assuming the PSF is Gaussian.
        If None is passed, class methods will assume the
        system is diffraction-limited with an Airy disk PSF
        and set psf_sigma to the diffraction limit
        (1.22*lambda*fnum).
    pix_scale: float
        The pixel scale of the observatory, in arcsec/pix.
    obs: pysynphot.observation object
        The Observation object combining all of the
        observatory's specifications.

    Methods
    ----------
    tot_signal(spectrum=None):
        Returns the signal for one exopsure of a source, in
        total number of electrons produced across the sensor.
    s_n(spectrum=None):
        Returns the S/N ratio for one stacked image.
    limiting_dist(spectrum=None):
        Returns the maximum distance, in pc, at which
        a given object can be detected. Accounts for
        dimming and redshift.
    limiting_mag():
        Returns the maximum AB magnitude that the
        observatory can detect in one stacked image.
    '''

    def __init__(
            self, sensor, telescope, filter_bandpass=1,
            exposure_time=1., num_exposures=1, eclip_lat=90,
            limiting_s_n=5., psf_sigma=None):
        '''Initialize Observatory class attributes.

        Parameters
        ----------
        sensor: Sensor object
            The sensor used in the observatory.
        telescope: Telescope object
            The telescope used in the observatory.
        filter_bandpass: pysynphot.bandpass object (default 1)
            The bandpass of the filter used. If not specified,
            assumes no filter is used.
        exposure_time: float (default 1.0)
            Duration for one exposure, in seconds.
        num_exposures: int (default 1)
            Number of exposures in a stacked image.
        limiting_s_n: float (default 5.0)
            The minimum signal to noise ratio for a detection.
        psf_sigma: float (default None)
            The standard deviation of the point spread function
            of the system, in um, assuming the PSF is Gaussian.
            If None is passed, assume the system is diffraction-
            limited with an Airy disk PSF and set psf_sigma to
            be the diffraction limit (1.22*lambda*fnum) within
            class methods.
        '''

        self.sensor = sensor
        self.telescope = telescope
        self.filter_bandpass = filter_bandpass
        self.bandpass = (filter_bandpass * self.telescope.bandpass *
                         self.sensor.qe)
        self.exposure_time = exposure_time
        self.num_exposures = num_exposures
        self.eclip_lat = eclip_lat
        self.limiting_s_n = limiting_s_n
        self.psf_sigma = psf_sigma

        plate_scale = 206265 / (self.telescope.focal_length * 10**4)
        self.pix_scale = plate_scale * self.sensor.pix_size
        self.dark_noise = self.sensor.dark_current * self.exposure_time
        self.bkg_noise = self.bkg_per_pix()

    def binset(self, spectrum):
        '''Narrowest binset from telescope, sensor, filter, and spectrum.'''
        binset_list = [self.filter_bandpass.wave, self.sensor.qe.wave,
                       self.telescope.bandpass.wave, spectrum.wave]
        binset_list = [x for x in binset_list if x is not None]
        range_list = [np.ptp(x) for x in binset_list]
        obs_binset = binset_list[np.argmin(range_list)]
        # Avoid bug when all throughputs are flat
        all_flat_q = (np.all((self.filter_bandpass.wave is None)) and
                      np.all((self.sensor.qe.wave is None)) and
                      np.all(self.telescope.bandpass.wave is None))
        if all_flat_q:
            array_bp = S.ArrayBandpass(obs_binset, np.ones(len(obs_binset)))
            self.bandpass = self.bandpass * array_bp
        return obs_binset
        # return spectrum.wave

    def tot_signal(self, spectrum):
        '''The total number of electrons generated in one exposure.

        Parameters
        ----------
        spectrum: pysynphot.spectrum object
            The spectrum for which to calculate the signal.
        '''
        S.setref(area=np.pi * self.telescope.diam ** 2/4)
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
        bkg_signal = self.tot_signal(bkg_sp)
        return bkg_signal

    def lambda_pivot(self):
        '''The pivot wavelength for observation of a given spectrum, in Ang.'''
        return self.bandpass.pivot()
    
    def psf_fwhm(self):
        '''The full width at half maximum of the PSF, in microns.'''
        if self.psf_sigma is None:
            fwhm = 1.025 * self.lambda_pivot() * self.telescope.f_num / 10 ** 4
        else:
            fwhm = 2.355 * self.psf_sigma
        return fwhm

    def single_pix_signal(self, spectrum):
        '''The signal within the central pixel of an image.

        Parameters
        ----------
        spectrum: pysynphot.spectrum object
            The spectrum for which to calculate the signal.
        '''

        pivot_wave = self.lambda_pivot()
        if self.psf_sigma is not None:
            half_width = self.sensor.pix_size / 2
            pix_frac = psfs.gaussian_ensq_energy(half_width, self.psf_sigma,
                                                 self.psf_sigma)
        else:
            half_width = (np.pi * self.sensor.pix_size /
                          (2 * self.telescope.f_num * pivot_wave * 10**-4))
            pix_frac = psfs.airy_ensq_energy(half_width)

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

    def limiting_dist(self, spectrum, initial_dist):
        '''Returns the max distance at which a source can be detected, in Mpc.

        Parameters
        ----------
        spectrum: pysynphot.spectrum object
            The spectrum for which to calculate the max distance.
        initial_dist: float
            The luminosity distance at which the spectrum is
            specified, in Mpc.
        '''

        ztab = RedshiftLookup()
        initial_sp = spectrum if spectrum is not None else self.spectrum

        def s_n_diff_dist(obs_dist):
            '''The difference between S/N at obs_dist and the limiting S/N.'''
            initial_z = ztab(initial_dist)
            obs_z = ztab(obs_dist)
            # Adjust the wavelengths of the source spectrum to account for
            # the redshift.
            obs_wave = initial_sp.wave * (1+initial_z) / (1+obs_z)
            obs_flux = (initial_sp.flux * (1+initial_z) / (1+obs_z)
                                        * (initial_dist/obs_dist)**2)
            obs_sp = S.ArraySpectrum(obs_wave, obs_flux,
                                     fluxunits=initial_sp.fluxunits)
            return self.single_pix_snr(obs_sp) - self.limiting_s_n

        # Newton-Raphson method for root-finding
        dist_tol, s_n_tol = 0.001, 0.01
        # Make an estimate for the limiting distance based on the initial S/N
        initial_s_n = s_n_diff_dist(initial_dist) + self.limiting_s_n
        dist_guess = initial_dist * np.sqrt(initial_s_n / self.limiting_s_n)
        dist = dist_guess
        dist_deriv_step = 0.01
        eps_dist = 10
        eps_s_n = s_n_diff_dist(dist)
        i = 1
        while abs(eps_s_n) > s_n_tol:
            if abs(eps_dist) < dist_tol:
                raise RuntimeError('No convergence to within 1 kpc.')
            elif i > 20:
                raise RuntimeError('No convergence after 20 iterations.')
            eps_s_n_prime = ((s_n_diff_dist(dist + dist_deriv_step) - eps_s_n)
                             / dist_deriv_step)
            eps_dist = eps_s_n / eps_s_n_prime
            dist -= eps_dist
            eps_s_n = s_n_diff_dist(dist)
            i += 1
        return dist

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
            return signal - self.sensor.full_well

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

    def avg_intensity_grid(self, spectrum, pos=np.array([0, 0]),
                           subarray_size=11, resolution=11):
        '''The average intensity across a subarray with sub-pixel resolution

        Parameters
        ----------
        spectrum: pysynphot.spectrum object
            The spectrum for which to calculate the intensity grid.
        pos: array-like (default np.array([0, 0]))
            The centroid position of the source on the subarray, in
            microns, with [0,0] representing the center of the
            central pixel.
        '''
        tot_signal = self.tot_signal(spectrum)
        # Simulate the PSF on a subarray of pixels, with sub-pixel resolution.
        if self.psf_sigma is not None:
            cov_mat = [[self.psf_sigma ** 2, 0], [0, self.psf_sigma ** 2]]
            psf_grid = psfs.gaussian_psf(subarray_size, resolution,
                                         self.sensor.pix_size, pos, cov_mat)
        else:
            psf_grid = psfs.airy_disk(subarray_size, resolution,
                                      self.sensor.pix_size, pos,
                                      self.telescope.f_num,
                                      self.lambda_pivot())
        intensity_grid = psf_grid * tot_signal
        return intensity_grid

    def obs_grid(self, spectrum, pos=np.array([0, 0]),
                 subarray_size=11, resolution=11):
        '''The intensity (in electrons) across a 9x9 pixel subarray

        Parameters
        ----------
        spectrum: pysynphot.spectrum object
            The spectrum for which to calculate the intensity grid.
        pos: array-like (default [0, 0])
            The centroid position of the source on the subarray, in
            microns, with [0,0] representing the center of the
            central pixel.
        '''
        base_grid = self.avg_intensity_grid(spectrum, pos, subarray_size,
                                            resolution)
        # Sum the signals within each pixel
        temp_grid = base_grid.reshape((11, resolution, 11, resolution))
        pixel_grid = temp_grid.sum(axis=(1, 3))
        return pixel_grid

    def observation(self, spectrum, pos=np.array([0, 0])):
        '''The optimal image, signal, and noise observed for a spectrum.

        Parameters
        ----------
        spectrum: pysynphot.spectrum object
            The spectrum for which to calculate the image.
        pos: array-like (default [0, 0])
            The centroid position of the source on the subarray, in
            microns, with [0,0] representing the center of the
            central pixel.

        Returns
        -------
        signal: float
            The total signal in the optimal aperture, in e-
        noise: float
            The total noise in the optimal aperture, in e-
        image: 2D array
            The image of the source, in e-/pix
        '''
        noise_per_pix = self.single_pix_noise()
        pixel_grid = self.obs_grid(spectrum, pos)
        # Determine the optimal aperture for the image
        optimal_ap = psfs.optimal_aperture(pixel_grid, noise_per_pix)
        n_aper = optimal_ap.sum()
        obs_grid = pixel_grid * optimal_ap
        frame_signal = obs_grid.sum()
        signal = frame_signal * self.num_exposures
        noise = np.sqrt(signal +
                        self.num_exposures * n_aper * noise_per_pix ** 2)
        return (signal, noise, obs_grid, optimal_ap)

    def snr(self, spectrum, pos=np.array([0, 0])):
        '''The snr of a given spectrum, using the optimal aperture.

        Parameters
        ----------
        spectrum: pysynphot.spectrum object
            The spectrum for which to calculate the snr.
        pos: array-like (default [0, 0])
            The centroid position of the source on the subarray, in
            microns, with [0,0] representing the center of the
            central pixel.

        Returns
        -------
        snr: float
            The signal to noise ratio for the spectrum.
        '''
        (signal, noise) = self.observation(spectrum, pos)[0:2]
        snr = signal / noise
        return snr


def blackbody_spec(temp, dist, l_bol):
    '''Returns a blackbody spectrum with the desired properties.

        Parameters
        ----------
        temp: float
            The temperature of the blackbody, in K.
        dist: float
            The luminosity distance at which the spectrum is
            specified, in Mpc.
        l_bol: float
            The bolometric luminosity of the source.
        '''
    # A default pysynphot blackbody is at 1 kpc and for a star with
    # radius r_sun
    spectrum = S.BlackBody(temp)
    ztab = RedshiftLookup()
    initial_z = ztab(10 ** -3)
    obs_z = ztab(dist)
    # Adjust the wavelengths of the source spectrum to account for
    # the redshift, and the flux for the luminosity distance.
    obs_wave = spectrum.wave * (1+initial_z) / (1+obs_z)
    obs_flux = (spectrum.flux * (1+initial_z) / (1+obs_z)
                * (10 ** -3 / dist) ** 2)
    # Scale the flux using the desired bolometric luminosity
    l_bol_scaling = l_bol / (4 * np.pi * constants.sigma *
                             constants.R_SUN ** 2 * temp ** 4)
    obs_flux *= l_bol_scaling
    obs_spectrum = S.ArraySpectrum(obs_wave, obs_flux,
                                   fluxunits=spectrum.fluxunits)
    return obs_spectrum

# Some tests of Observatory behavior, using a spectrum similar to the sun.
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

    flat_spec = S.FlatSpectrum(25, fluxunits='abmag')
    flat_spec.convert('fnu')

    # tess_geo_v = Observatory(telescope=mono_tele_v10, sensor=imx455,
    #                          filter_bandpass=v_bandpass,
    #                          exposure_time=300, num_exposures=3)

    tess_geo_obs = Observatory(telescope=mono_tele_v10, sensor=imx455,
                               filter_bandpass=vis_bandpass, eclip_lat=90,
                               exposure_time=300, num_exposures=3)

    # tess_geo_b = Observatory(telescope=mono_tele_v10, sensor=imx455,
    #                          filter_bandpass=b_bandpass, eclip_lat=90,
    #                          exposure_time=300, num_exposures=3)

    print(tess_geo_obs.limiting_mag())
