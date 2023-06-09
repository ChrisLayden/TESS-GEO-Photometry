# Chris Layden

'''Defining sensors and telescopes used in TESS-GEO.'''

import os
import pysynphot as S
import numpy as np
from observatory import Sensor, Telescope, Observatory

data_folder = os.path.dirname(__file__) + '/../data/'

# Defining sensors
# Sensor dark currents given at -25 deg C, extrapolated from QHY600 data
imx455_qe = S.FileBandpass(data_folder + 'imx455.fits')
imx455 = Sensor(pix_size=3.76, read_noise=1.65, dark_current=1.5*10**-3,
                full_well=51000, qe=imx455_qe)

imx487_qe = S.FileBandpass(data_folder + 'imx487.fits')
imx487 = Sensor(pix_size=2.74, read_noise=3, dark_current=5**-4,
                full_well=100000, qe=imx487_qe)

ultrasat_arr = np.genfromtxt(data_folder + 'ULTRASAT_QE.csv', delimiter=',')
ultrasat_qe = S.ArrayBandpass(ultrasat_arr[:, 0], ultrasat_arr[:, 1])
ultrasat_cmos = Sensor(pix_size=9.5, read_noise=3.5, dark_current=0.026,
                       full_well=100000, qe=ultrasat_qe)

tess_qe = S.FileBandpass(data_folder + 'tess.fits')
# This dark current is just a place holder; it's negligible anyways
tesscam = Sensor(pix_size=15, read_noise=10, dark_current=5**-4,
                 full_well=200000, qe=tess_qe)

sensor_dict = {'IMX 455 (Visible)': imx455, 'IMX 487 (UV)': imx487,
               'TESS CCD': tesscam, 'ULTRASAT CMOS': ultrasat_cmos}

# Defining telescopes
v10_bandpass = S.UniformTransmission(0.693)
mono_tele_v10 = Telescope(diam=25, f_num=8, bandpass=v10_bandpass)

v11_bandpass = S.UniformTransmission(0.54)
mono_tele_v11 = Telescope(diam=28.5, f_num=3.5, bandpass=v11_bandpass)

v3_bandpass = S.UniformTransmission(0.54)
mono_tele_v3 = Telescope(diam=8.5, f_num=3.5, bandpass=v3_bandpass)

# Transmission is scaled to give 15,000 e-/s from a mag 10 star
tess_tele_thru = S.UniformTransmission(0.6315)
tess_tele = Telescope(diam=10.5, f_num=1.4, bandpass=tess_tele_thru)


data_folder = os.path.dirname(__file__) + '/../data/'
uv_filter = S.FileBandpass(data_folder + 'uv_200_300.fits')
AIRY_FWHM = 1.025 * 2500 * 3.5 / 10 ** 4
UV_SIGMA = 2 * AIRY_FWHM / 2.355
tess_geo_v3 = Observatory(imx487, mono_tele_v3, filter_bandpass=uv_filter,
                          exposure_time=1, num_exposures=1,
                          psf_sigma=UV_SIGMA)
flat_spec = S.FlatSpectrum(16.6, fluxunits='abmag')

tess_geo_v11 = Observatory(imx487, mono_tele_v11, filter_bandpass=uv_filter,
                           exposure_time=60, num_exposures=1,
                           psf_sigma=UV_SIGMA)

telescope_dict = {'Mono Tele V10 (Visible)': mono_tele_v10,
                  'Mono Tele V11 (UV)': mono_tele_v11,
                  'Mono Tele V3 (UV)': mono_tele_v3,
                  'TESS Telescope (IR)': tess_tele}

# Defining filters
no_filter = S.UniformTransmission(1)
johnson_u = S.ObsBandpass('johnson,u')
johnson_b = S.ObsBandpass('johnson,b')
johnson_v = S.ObsBandpass('johnson,v')
johnson_r = S.ObsBandpass('johnson,r')
johnson_i = S.ObsBandpass('johnson,i')
uv_80 = S.FileBandpass(data_folder + 'uv_200_300.fits')

ultrasat_filt_arr = np.genfromtxt(data_folder + 'ULTRASAT_Filter.csv',
                                  delimiter=',')
ultrasat_filter = S.ArrayBandpass(ultrasat_filt_arr[:, 0],
                                  ultrasat_filt_arr[:, 1] / 100)

filter_dict = {'None': no_filter, 'Johnson U': johnson_u,
               'Johnson B': johnson_b, 'Johnson V': johnson_v,
               'Johnson R': johnson_r, 'Johnson I': johnson_i,
               '200-300 nm 80%': uv_80, 'ULTRASAT': ultrasat_filter}
