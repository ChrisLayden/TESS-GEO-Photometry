'''Functions for calculating PSFs and optimal apertures.

Functions
---------
airy_ensq_energy : float
    The fraction of the light that hits a square of half-width p
    centered on an Airy disk PSF.
gaussian_ensq_energy : float
    The fraction of the light that hits a square of half-width p
    centered on a Gaussian PSF.
gaussian_psf : array-like
    An x-y grid with a Gaussian disk evaluated at each point.
airy_disk : array-like
    An x-y grid with the Airy disk evaluated at each point.
optimal_aperture : array-like
    The optimal aperture for maximizing S/N.
'''

import numpy as np
from scipy import special, integrate
from scipy.signal import fftconvolve


# Calculate the energy in a square of dimensionless half-width p
# centered on an Airy disk PSF. From Eq. 7 in Torben Anderson's paper,
# Vol. 54, No. 25 / September 1 2015 / Applied Optics
# http://dx.doi.org/10.1364/AO.54.007525
def airy_ensq_energy(half_width):
    '''Returns the energy in a square of half-width p centered on an Airy PSF.

    Parameters
    ----------
    half_width : float
        The half-width of the square, defined in the paper linked above.

    Returns
    -------
    pix_fraction : float
        The fraction of the light that hits the square.
    '''
    def ensq_int(theta):
        '''Integrand to calculate the ensquared energy'''
        return 4/np.pi * (1 - special.jv(0, half_width/np.cos(theta))**2
                            - special.jv(1, half_width/np.cos(theta))**2)
    pix_fraction = integrate.quad(ensq_int, 0, np.pi/4)[0]
    return pix_fraction


# Calculate the energy in a square of half-width p (in um)
# centered on an Gaussian PSF with x and y standard deviations
# sigma_x and sigma_y, respectively. Currently doesn't let you
# have any covariance between x and y.
def gaussian_ensq_energy(half_width, sigma_x, sigma_y):
    '''Returns the energy in square of half-width p centered on a Gaussian PSF.

    Parameters
    ----------
    half_width : float
        The half-width of the square, in units of um.
    sigma_x : float
        The standard deviation of the Gaussian in the x direction, in um.
    sigma_y : float
        The standard deviation of the Gaussian in the y direction, in um.

    Returns
    -------
    pix_fraction : float
        The fraction of the light that hits the square.
    '''
    arg_x = half_width / np.sqrt(2) / sigma_x
    arg_y = half_width / np.sqrt(2) / sigma_y
    pix_fraction = special.erf(arg_x) * special.erf(arg_y)
    return pix_fraction


def gaussian_psf(num_pix, resolution, pix_size, mu, Sigma):
    '''Return an x-y grid with a Gaussian disk evaluated at each point.

    Parameters
    ----------
    num_pix : int
        The number of pixels in the subarray.
    resolution : int
        The number of subpixels per pixel in the subarray.
    pix_size : float
        The size of each pixel in the subarray, in microns.
    mu : array-like
        The mean of the Gaussian, in microns.
    Sigma : array-like
        The covariance matrix of the Gaussian, in microns.

    Returns
    -------
    gaussian : array-like
        The Gaussian PSF evaluated at each point in the subarray,
        normalized to have a total amplitude of the fractional energy
        ensquared in the subarray.
    '''
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
    # Normalize the PSF to have a total amplitude of subarray_fraction
    gaussian = np.exp(-arg / 2)
    normalize = subarray_fraction / gaussian.sum()
    return gaussian * normalize


def airy_disk(num_pix, resolution, pix_size, mu, fnum, lam):
    '''Return an x-y grid with the Airy disk evaluated at each point.

    Parameters
    ----------
    num_pix : int
        The number of pixels in the subarray.
    resolution : int
        The number of subpixels per pixel in the subarray.
    pix_size : float
        The size of each pixel in the subarray, in microns.
    mu : array-like
        The mean position of the Airy disk, in pixels.
    fnum : float
        The f-number of the telescope.
    lam : float
        The wavelength of the light, in Angstroms.

    Returns
    -------
    airy : array-like
        The Airy disk evaluated at each point in the subarray,
        normalized to have a total amplitude of the fractional energy
        ensquared in the subarray.
    '''
    lam /= 10 ** 4
    grid_points = num_pix * resolution
    x = np.linspace(-num_pix / 2, num_pix / 2, grid_points) * pix_size
    y = np.linspace(-num_pix / 2, num_pix / 2, grid_points) * pix_size
    x, y = np.meshgrid(x, y)
    pos = np.empty(x.shape + (1,))
    pos[:, :, 0] = np.sqrt((x - mu[0]) ** 2 + (y - mu[1]) ** 2)
    pos = pos[:, :, 0]
    arg = np.pi / (lam) / fnum * pos
    # Avoid singularity at origin
    arg[arg == 0] = 10 ** -10
    airy = (special.jv(1, arg) / arg) ** 2 / np.pi
    # Determine the fraction of the light that hits the entire subarray
    array_p = num_pix / 2 * pix_size * np.pi / fnum / lam
    subarray_fraction = airy_ensq_energy(array_p)
    # Normalize the PSF to have a total amplitude of subarray_fraction
    normalize = subarray_fraction / airy.sum()
    return airy * normalize

def moffat_psf(num_pix, resolution, pix_size, mu, alpha, beta):
    '''Return an x-y grid with a Moffat distribution evaluated at each point.

    Parameters
    ----------
    num_pix : int
        The number of pixels in the subarray.
    resolution : int
        The number of subpixels per pixel in the subarray.
    pix_size : float
        The size of each pixel in the subarray, in microns.
    mu : array-like
        The mean of the Gaussian, in pixels.
    alpha: float
        The width of the Moffat distribution, in microns.
    beta: float
        The power of the Moffat distribution.

    Returns
    -------
    gaussian : array-like
        The Gaussian PSF evaluated at each point in the subarray,
        normalized to have a total amplitude of the fractional energy
        ensquared in the subarray.
    '''
    grid_points = num_pix * resolution
    x = np.linspace(-num_pix / 2, num_pix / 2, grid_points) * pix_size
    y = np.linspace(-num_pix / 2, num_pix / 2, grid_points) * pix_size
    x, y = np.meshgrid(x, y)
    pos = np.empty(x.shape + (1,))
    pos[:, :, 0] = 1 + np.sqrt((x - mu[0]) ** 2 + (y - mu[1]) ** 2) / alpha ** 2
    arg = (beta - 1) / (np.pi * alpha ** 2) * (1 + pos[:, :, 0]) ** -beta
    return arg
    # This isn't done yet--need to do normalization

def jittered_psf(psf_subgrid, pix_jitter, resolution):
    '''Convolve the jitter profile with the PSF.
    
    Parameters
    ----------
    psf_subgrid : array-like
        The PSF subgrid.
    pix_jitter : float
        The RMS jitter, in pixels.
    resolution : int
        The number of subpixels per pixel in the subgrid.
    
    Returns
    -------
    jittered_psf : array-like
        The jittered PSF.
    '''
    # Generate a 2D Gaussian array with sigma equal to pix_jitter
    num_pix = int(psf_subgrid.shape[0] / resolution)
    jitter_profile = gaussian_psf(num_pix, resolution, 1, [0, 0],
                                  [[pix_jitter, 0], [0, pix_jitter]])
    jittered_psf = fftconvolve(psf_subgrid, jitter_profile, mode='same')
    return jittered_psf

def optimal_aperture(psf_grid, noise_per_pix):
    '''The optimal aperture for maximizing S/N.

    Parameters
    ----------
    psf_grid: array-like
        The signal recorded in each pixel
    noise_per_pix: float
        The noise per pixel, besides source shot noise.

    Returns
    -------
    aperture_grid: array-like
        A grid of 1s and 0s, where 1s indicate pixels that should be
        included in the optimal aperture.
    '''

    # Copy the image to a new array so we aren't modifying
    # the original array
    func_grid = psf_grid.copy()
    aperture_grid = np.zeros(psf_grid.shape)
    n_aper = 0
    signal = 0
    snr_max = 0

    while n_aper <= func_grid.size:
        imax, jmax = np.unravel_index(func_grid.argmax(), func_grid.shape)
        Nmax = psf_grid[imax, jmax]
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

if __name__ == '__main__':
    # Test the functions
    import matplotlib.pyplot as plt

    # # Test the Gaussian
    # gaussian = gaussian_psf(5, 5, 1, [0,0], [[1,0],[0,1]])
    # fig, ax = plt.subplots()
    # im = ax.imshow(gaussian)
    # plt.colorbar(im)
    # plt.show()

    # Test the Airy
    airy = airy_disk(10, 10, 1, [0,0], 5, 5000)
    # Test the jittering
    jittered = jittered_psf(airy, 0.5, 10)

    # Plot them next to each other
    fig, (ax1, ax2) = plt.subplots(1, 2)
    im1 = ax1.imshow(np.log(airy))
    plt.colorbar(im1, ax=ax1)
    im2 = ax2.imshow(np.log(jittered))
    plt.colorbar(im2, ax=ax2)
    plt.show()

    # # Test the Moffat
    # moffat = moffat_psf(10, 10, 1, [0,0], 5, 12.5)
    # fig, ax = plt.subplots()
    # im = ax.imshow(np.log(moffat))
    # plt.colorbar(im)
    # plt.show()
