import pysynphot as S
from redshift_lookup import RedshiftLookup
import numpy as np
import constants


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


power_law_1 = S.PowerLaw(5000, -1)
