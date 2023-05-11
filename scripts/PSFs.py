# Chris Layden

"""Functions useful for simulating and analyzing PSFs."""

import numpy as np
from scipy import special, integrate

# Calculate the energy in a square of dimensionless half-width p
# centered on an Airy disk PSF. From Eq. 7 in Torben Anderson's paper,
# Vol. 54, No. 25 / September 1 2015 / Applied Optics
# http://dx.doi.org/10.1364/AO.54.007525
def airy_ensq_energy(p):
    def ensq_int(theta):
        """Integrand to calculate the ensquared energy"""
        return 4/np.pi * (1 - special.jv(0, p/np.cos(theta))**2
                            - special.jv(1, p/np.cos(theta))**2)
    pix_fraction = integrate.quad(ensq_int, 0, np.pi/4)[0]
    return pix_fraction

# Calculate the energy in a square of half-width p (in um)
# centered on an Gaussian PSF with x and y standard deviations
# sigma_x and sigma_y, respectively.
def gaussian_ensq_energy(p, sigma_x, sigma_y):
    arg_x = p / np.sqrt(2) / sigma_x
    arg_y = p / np.sqrt(2) / sigma_y
    pix_fraction = special.erf(arg_x) * special.erf(arg_y)
    return pix_fraction


def gaussian_psf(num_pix, resolution, pix_size, mu, Sigma):
    """Return an x-y grid with a Gaussian disk evaluated at each point."""
    grid_points = num_pix * resolution
    x = np.linspace(-num_pix / 2, num_pix / 2, grid_points) * pix_size
    y = np.linspace(-num_pix / 2, num_pix / 2, grid_points) * pix_size
    x, y = np.meshgrid(x, y)
    pos = np.empty(x.shape + (2,))
    pos[:, :, 0] = x
    pos[:, :, 1] = y

    Sigma_inv = np.linalg.inv(Sigma)
    # This einsum call calculates (x-mu)T.Sigma-1.(x-mu) in a vectorized
    # way across all the input variables.
    arg = np.einsum('...k,kl,...l->...', pos-mu, Sigma_inv, pos-mu)
    # Determine the fraction of the light that hits the entire subarray
    array_p = num_pix / 2 * pix_size
    subarray_fraction = gaussian_ensq_energy(array_p, Sigma[0][0], Sigma[1][1])
    # Normalize the PSF to have a total amplitude of 1
    gaussian = np.exp(-arg / 2)
    normalize = subarray_fraction / gaussian.sum()
    return gaussian * normalize

def airy_disk(num_pix, resolution, pix_size, mu, fnum, lam):
    """Return an x-y grid with the Airy disk evaluated at each point."""
    # pix_size in microns
    # lam in angstrom
    lam /= 10 ** 4
    grid_points = num_pix * resolution
    x = np.linspace(-num_pix / 2, num_pix / 2, grid_points) * pix_size
    y = np.linspace(-num_pix / 2, num_pix / 2, grid_points) * pix_size
    x, y = np.meshgrid(x, y)
    pos = np.empty(x.shape + (1,))
    pos[:, :, 0] = np.sqrt((x - mu[0]) ** 2 + (y - mu[1]) ** 2)
    
    # Make sure to convert lam from angstroms to um
    arg = np.pi / (lam) / fnum * pos
    # Avoid singularity at origin
    arg[arg==0] = 10 ** -10
    airy = (special.jv(1, arg) / arg) ** 2 / np.pi
    # Determine the fraction of the light that hits the entire subarray
    array_p = num_pix / 2 * pix_size * np.pi / fnum / lam
    subarray_fraction = airy_ensq_energy(array_p)
    # Normalize the PSF to have a total amplitude of 1
    normalize = subarray_fraction / airy.sum()
    return airy * normalize

# Finding the aperture that maximizes the SNR for an image. Uses the
# algorithm detailed in Tam Nguyen's thesis.
def optimal_aperture(prf_grid, noise_per_pix):
    """The optimal aperture for maximizing S/N.

    Parameters
    ----------
    prf_grid: array-like
        The signal recorded in each pixel
    noise_per_pix: float
        The noise per pixel, besides source shot noise.
    """

    # Copy the image to a new array so we aren't modifying
    # the original array
    func_grid = prf_grid.copy()
    aperture_grid = np.zeros(prf_grid.shape)
    n_aper = 0
    signal = 0
    snr_max = 0

    while n_aper <= func_grid.size:
        imax, jmax = np.unravel_index(func_grid.argmax(), func_grid.shape)
        Nmax = prf_grid[imax, jmax]
        func_grid[imax, jmax] = -1

        signal = signal + Nmax
        noise = np.sqrt(signal + ((n_aper + 1) * noise_per_pix) ** 2)
        snr = signal / noise

        if snr > snr_max:
            snr_max = snr
            aperture_grid[imax, jmax] = 1
            n_aper += 1
        else:
            break

    return aperture_grid
