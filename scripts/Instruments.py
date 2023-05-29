# Chris Layden

"""Defining sensors and telescopes used in TESS-GEO.

Sensor Objects
----------
imx455
imx487

Telescope Objects
----------
mono_tele_v10
mono_tele_v11
"""

import os
from Observatory import Sensor, Telescope, Observatory
import pysynphot as S

data_folder = os.path.dirname(__file__) + '/../data/'

# Sensor dark currents given at -25 deg C, extrapolated from QHY600 data
imx455_qe = S.FileBandpass(data_folder + 'imx455.fits')
imx455 = Sensor(pix_size=3.76, read_noise=1.65, dark_current=1.5*10**-3,
                full_well=51000, qe=imx455_qe)

imx487_qe = S.FileBandpass(data_folder + 'imx487.fits')
imx487 = Sensor(pix_size=2.74, read_noise=3, dark_current=5**-4,
                full_well=100000, qe=imx487_qe)

v10_bandpass = S.UniformTransmission(0.693)
mono_tele_v10 = Telescope(diam=25, f_num=8, bandpass=v10_bandpass)

v11_bandpass = S.UniformTransmission(0.54)
mono_tele_v11 = Telescope(diam=28.5, f_num=5, bandpass=v11_bandpass)


data_folder = os.path.dirname(__file__) + '/../data/'
uv_filter = S.FileBandpass(data_folder + "uv_200_300.fits")
airy_fwhm = 1.025 * 2500 * 5/  10 ** 4
uv_sigma = 2 * airy_fwhm / 2.355
# print(uv_sigma)
tess_geo_uv = Observatory(imx487, mono_tele_v11, filter_bandpass=uv_filter, exposure_time=1,
                          num_exposures=1, psf_sigma=uv_sigma)