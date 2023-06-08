import numpy as np
import os

abs_path = os.path.dirname(__file__)
# Log(specific intensity) of the zodiacal light at ecliptic latitude 90 deg,
eclip_ilam = np.genfromtxt(abs_path + '/../data/ZodiacalLight.csv', delimiter=',')
eclip_ilam[:,1] = 10 ** eclip_ilam[:,1]
# The specific intensity for a V-band baseline
eclip_ilam_v = np.interp(5500, eclip_ilam[:,0], eclip_ilam[:,1])

def bkg_Ilam(lam, b):
    '''Return the specific intensity of sky background light.
    
    Parameters
    ----------
    lam : float
        The wavelength of the light, in Angstroms.
    b : float
        The ecliptic latitude, in degrees. Note we assume the specific intensity
        scales with b in the same way as it does for zodiacal light in the
        V-band. This is conservative for most other bands, especially the UV,
        for which most background light comes from diffuse galactic light.
    Returns
    -------
    ilam : float
        The specific intensity of the sky background, in erg/s/cm^2/Ang/arcsec^2.
    '''
    # Uses a linear fit to the magnitude of zodiacal light in the V-band as a
    # function of ecliptic latitude described in Sullivan et al. 2015
    vmag_max = 23.345
    del_vmag = 1.148
    # The V-band magnitude, in mag/arcsec^2
    vmag = vmag_max - del_vmag * ((b - 90) / 90) ** 2
    # The V-band specific intensity, in erg/s/cm^2/Hz/arcsec^2
    inu_v = 10 ** (-vmag / 2.5) * 3631 * 10 ** -23
    # Make sure to use c in Angstroms
    ilam_v = inu_v * (3 * 10 ** 18) / lam ** 2
    freq_factor = np.interp(lam, eclip_ilam[:,0], eclip_ilam[:,1]) / eclip_ilam_v
    ilam = ilam_v * freq_factor
    return ilam

def bkg_spectrum(b):
    '''Return the spectrum of light from the sky background.
    
    Parameters
    ----------
    b : float
        The ecliptic latitude, in degrees.
    Returns
    -------
    spectrum : array-like
        The background spectrum, in erg/s/cm^2/Ang/arcsec^2.
    '''
    # The wavelengths, in Angstroms
    lam = eclip_ilam[:,0]
    # The specific intensity, in erg/s/cm^2/Ang/arcsec^2
    ilam = bkg_Ilam(lam, b)
    return np.array([lam, ilam])
