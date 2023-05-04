# Chris Layden

"""Classes and functions for synthetic photometry and noise characterization.

Classes
----------
Sensor
Telescope
Observatory
"""
import matplotlib.pyplot as plt
import os
from scipy import special
from scipy import integrate
import numpy as np
import pysynphot as S
from RedshiftLookup import RedshiftLookup
from constants import *
from PSFs import *

pysynphot_path = os.environ['PYSYN_CDBS']


class Sensor(object):
    """Class specifying a photon-counting sensor.

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
    """

    def __init__(self, pix_size, read_noise, dark_current, qe=1):
        """Initialize a Sensor object.

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
        """

        self.pix_size = pix_size
        self.read_noise = read_noise
        self.dark_current = dark_current
        self.qe = qe


class Telescope(object):
    """Class specifying a telescope.

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
    """
    def __init__(self, diam, f_num, bandpass=1):
        """Initializing a telescope object.

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

        """

        self.diam = diam
        self.f_num = f_num
        self.bandpass = bandpass
        self.focal_length = self.diam * self.f_num
        self.plate_scale = 206265 / (self.focal_length * 10**4)


class Observatory(object):
    """Class specifying a complete observatory.

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
    num_pix: int
        The number of pixels used to define an image
        of one point source.
    psf_sigma: float CURRENTLY UNUSED
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
    background: float
        The signal from the background in one exposure, in
        electrons.

    Methods
    ----------
    tot_signal(spectrum=None):
        Returns the signal for one exopsure of a source, in
        total number of electrons produced across the sensor.
    noise(spectrum=None, verbose=False, return_signal=False):
        Returns the noise for one exposure of a source.
    s_n(spectrum=None):
        Returns the S/N ratio for one stacked image.
    limiting_dist(spectrum=None):
        Returns the maximum distance, in pc, at which
        a given object can be detected. Accounts for
        dimming and redshift.
    limiting_mag():
        Returns the maximum AB magnitude that the
        observatory can detect in one stacked image.
    """

    def __init__(
            self, sensor, telescope, filter_bandpass=1,
            exposure_time=1., num_exposures=1, limiting_s_n=5.,
            num_pix=4, psf_sigma=None):
        """Initialize Observatory class attributes.

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
        num_pix: int (default 4)
            The number of pixels used to define an image
            of one point source.
        psf_sigma: float (default None)
            The standard deviation of the point spread function
            of the system, in um, assuming the PSF is Gaussian.
            If None is passed, assume the system is diffraction-
            limited with an Airy disk PSF and set psf_sigma to
            be the diffraction limit (1.22*lambda*fnum) within
            class methods.
        """

        self.sensor = sensor
        self.telescope = telescope
        self.bandpass = (filter_bandpass * self.telescope.bandpass *
                         self.sensor.qe)
        self.exposure_time = exposure_time
        self.num_exposures = num_exposures
        self.limiting_s_n = limiting_s_n
        self.num_pix = num_pix
        self.psf_sigma = psf_sigma

        plate_scale = 206265 / (self.telescope.focal_length * 10**4)
        self.pix_scale = plate_scale * self.sensor.pix_size

        S.setref(area=np.pi * self.telescope.diam ** 2/4)

    def tot_signal(self, spectrum):
        """The total number of electrons generated in one exposure.

        Parameters
        ----------
        spectrum: pysynphot.spectrum object
            The spectrum for which to calculate the signal.
        """
        obs = S.Observation(spectrum, self.bandpass, binset=spectrum.wave,
                            force='extrap')
        raw_rate = obs.countrate()
        signal = raw_rate * self.exposure_time
        return signal

    def bkg_per_pix(self):
        """The signal generated by the background in one exposure and pixel."""
        # Array of wavelengths at which the background specific flux is known
        bkg_array_lam = np.array([
            1485, 2000, 2504, 3019, 3611,
            4005, 4400, 5496, 7414, 8400
            ])
        # Array of intensities of the background, in erg/s/cm^2/Ang/arcsec^2
        bkg_array_spec_flam = np.array([
            6.50681e-18, 1.08333e-18, 1.20103e-18, 2.71078e-18, 3.00307e-18,
            4.24244e-18, 3.81870e-18, 2.44753e-18, 9.47797e-19, 6.08502e-19
            ])
        # Multiply specific flux by solid angle subtended by one pixel
        bkg_array_flam = bkg_array_spec_flam * self.pix_scale**2
        bkg_sp = S.ArraySpectrum(bkg_array_lam, bkg_array_flam,
                                 fluxunits='flam')
        bkg_signal = self.tot_signal(bkg_sp)
        return bkg_signal

    def lambda_pivot(self, spectrum):
        """The pivot wavelength for observation of a given spectrum."""
        obs = S.Observation(spectrum, self.bandpass, binset=spectrum.wave,
                            force='extrap')
        return obs.pivot()

    def single_pix_signal(self, spectrum):
        """The signal within the central pixel of an image.

        Parameters
        ----------
        spectrum: pysynphot.spectrum object
            The spectrum for which to calculate the signal.
        """

        pivot_wave = self.lambda_pivot(spectrum)
        if self.psf_sigma is not None:
            p = self.sensor.pix_size / 2
            pix_fraction = gaussian_ensq_energy(p, self.psf_sigma)
        else:
            p = (np.pi * self.sensor.pix_size /
                   (2 * self.telescope.f_num * pivot_wave * 10**-4))
            pix_fraction = airy_ensq_energy(p)
        
        signal = self.tot_signal(spectrum) * pix_fraction
        return signal

    def single_pix_noise(self):
        """The noise from the background and sensor, in e-/pix.
        """
        # Noise from sensor dark current
        dark_current_noise = np.sqrt(self.sensor.dark_current *
                                     self.exposure_time)
        # Shot noise from the background
        bkg_noise = np.sqrt(self.bkg_per_pix())

        # Add noise in quadrature
        noise = np.sqrt(dark_current_noise ** 2 +
                        self.sensor.read_noise ** 2 + bkg_noise ** 2
                        )

        return noise

    def single_pix_snr(self, spectrum):
        """The SNR for a given source with a single-pixel aperature.

        Parameters
        ----------
        spectrum: pysynphot.spectrum object
            The spectrum for which to calculate the SNR.
        """

        signal = self.single_pix_signal(spectrum)
        noise = np.sqrt(signal + self.single_pix_noise() ** 2)
        exposure_snr = signal / noise
        stack_snr = exposure_snr * np.sqrt(self.num_exposures)
        print(signal, noise, stack_snr)
        return stack_snr

    def limiting_dist(self, spectrum, initial_dist):
        """Returns the max distance at which a source can be detected, in Mpc.

        Parameters
        ----------
        spectrum: pysynphot.spectrum object
            The spectrum for which to calculate the max distance.
        initial_dist: float
            The luminosity distance at which the spectrum is
            specified, in Mpc.
        """

        ztab = RedshiftLookup()
        initial_sp = spectrum if spectrum is not None else self.spectrum

        def s_n_diff_dist(obs_dist):
            """The difference between S/N at obs_dist and the limiting S/N."""
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
        """The limiting AB magnitude for the observatory."""
        # We use an aperture of just 1 pixel, as this is the optimal
        # aperture for very dark objects.
        mag_10_spectrum = S.FlatSpectrum(10, fluxunits='abmag')
        mag_10_spectrum.convert('fnu')
        mag_10_signal = self.single_pix_signal(mag_10_spectrum)
        pix_noise = self.single_pix_noise()

        def s_n_diff_mag(mag):
            """The difference between the S/N at mag and the limiting S/N."""
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

    def snr(self, spectrum):
        tot_signal = self.tot_signal(spectrum)
        noise_per_pix = self.single_pix_noise()
        # Simulate the PSF on a subarray of 9x9 pixels, with a resolution of 
        # 101x101 points per pixel.
        if self.psf_sigma is not None:
            base_grid = multivariate_gaussian(9, 101, self.sensor.pix_size,
                                              [0,0], [[self.psf_sigma,0],[0,self.psf_sigma]])
        else:
            base_grid = airy_disk(9, 101, self.sensor.pix_size, [0,0], self.telescope.f_num,
                                  self.lambda_pivot(spectrum))
        base_grid *= tot_signal
        # Sum the signals within each pixel
        temp_grid = base_grid.reshape((9, 101, 9, 101))
        pixel_grid = temp_grid.sum(axis=(1, 3))
        # Determine the optimal aperture for the image
        optimal_ap, n_aper = optimal_aperature(pixel_grid, noise_per_pix)
        obs_grid = pixel_grid * optimal_ap
        signal = obs_grid.sum()
        noise =  np.sqrt(signal + (n_aper * noise_per_pix) ** 2)
        snr = signal / noise * np.sqrt(self.num_exposures)
        return snr, obs_grid

def blackbody_spec(temp, dist, l_bol):
    """Returns a blackbody spectrum with the desired properties.

        Parameters
        ----------
        temp: float
            The temperature of the blackbody, in K.
        dist: float
            The luminosity distance at which the spectrum is
            specified, in Mpc.
        l_bol: float
            The bolometric luminosity of the source.
        """
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
    l_bol_scaling = l_bol / (4 * np.pi * sigma * R_SUN ** 2 * temp ** 4)
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
                    qe=sensor_bandpass)

    mono_tele_v10 = Telescope(diam=25, f_num=8, bandpass=telescope_bandpass)
    # vis_bandpass = S.ObsBandpass('johnson,b')
    vis_bandpass = S.UniformTransmission(1.0)

    tess_geo_obs = Observatory(telescope=mono_tele_v10, sensor=imx455,
                               filter_bandpass=vis_bandpass,
                               exposure_time=300, num_exposures=3)

    flat_spec = S.FlatSpectrum(25, fluxunits='abmag')
    flat_spec.convert('fnu')
    tess_geo_obs.single_pix_snr(flat_spec)
    test, ap = tess_geo_obs.snr(flat_spec)
