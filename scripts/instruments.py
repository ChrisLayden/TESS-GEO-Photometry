# Chris Layden

'''Defining sensors and telescopes used in TESS-GEO.'''

import os
import pysynphot as S
import numpy as np
from observatory import Sensor, Telescope

data_folder = os.path.dirname(__file__) + '/../data/'

# Defining sensors
# Sensor dark currents given at -25 deg C, extrapolated from QHY600 data
imx455_qe = S.FileBandpass(data_folder + 'imx455.fits')
imx455 = Sensor(pix_size=3.76, read_noise=1.65, dark_current=1.5*10**-3,
                full_well=51000, qe=imx455_qe)


imx487_qe = S.FileBandpass(data_folder + 'imx487.fits')
imx487 = Sensor(pix_size=2.74, read_noise=2.51, dark_current=5**-4,
                full_well=100000, qe=imx487_qe)

ultrasat_arr = np.genfromtxt(data_folder + 'ULTRASAT_QE.csv', delimiter=',')
ultrasat_qe = S.ArrayBandpass(ultrasat_arr[:, 0], ultrasat_arr[:, 1])
ultrasat_cmos = Sensor(pix_size=9.5, read_noise=3.5, dark_current=0.026,
                       full_well=100000, qe=ultrasat_qe)

imx990_arr = np.genfromtxt(data_folder + 'imx990_QE.csv', delimiter=',')
# Multiply the first column by 10 to convert from nm to Angstroms
imx990_arr[:, 0] *= 10
imx990_qe = S.ArrayBandpass(imx990_arr[:, 0], imx990_arr[:, 1])
# Lowest gain mode at -30 deg C
imx990_low_gain = Sensor(pix_size=5, read_noise=150, dark_current=47.7,
                         full_well=120000, qe=imx990_qe)
# Highest gain mode at -60 deg C
imx990 = Sensor(pix_size=5, read_noise=20, dark_current=10,
                full_well=2000, qe=imx990_qe)

tess_qe = S.FileBandpass(data_folder + 'tess.fits')
# This dark current is just a place holder; it's negligible anyways
tesscam = Sensor(pix_size=15, read_noise=10, dark_current=5**-4,
                 full_well=200000, qe=tess_qe)

sensor_dict = {'IMX 455 (Visible)': imx455, 'IMX 487 (UV)': imx487,
               'TESS CCD': tesscam, 'ULTRASAT CMOS': ultrasat_cmos,
               'IMX 990 (SWIR)': imx990}

# Defining telescopes
v10_bandpass = S.UniformTransmission(0.693)
mono_tele_v10 = Telescope(diam=25, f_num=8, bandpass=v10_bandpass)

mono_tele_v8_uv = Telescope(diam=17.5, f_num=4.5, bandpass=S.UniformTransmission(0.638))
mono_tele_v8_vis = Telescope(diam=17.5, f_num=4.5, bandpass=S.UniformTransmission(0.758))

# V10 UVS telescope with visible coatings
mono_tele_v10_vis = Telescope(diam=25, f_num=4.8, bandpass=S.UniformTransmission(0.758))
# V10 UVS telescope with UV coatings
mono_tele_v10_uv = Telescope(diam=25, f_num=4.8, bandpass=S.UniformTransmission(0.638))

mono_tele_v20_vis = Telescope(diam=47, f_num=4.8, bandpass=S.UniformTransmission(0.758))

v3uv_bandpass = S.UniformTransmission(0.54)
v3swir_bandpass = S.UniformTransmission(0.54*0.95/0.8)
mono_tele_v3uv = Telescope(diam=8.5, f_num=3.6, bandpass=v3uv_bandpass)
mono_tele_v3swir = Telescope(diam=8.5, f_num=3.6, bandpass=v3swir_bandpass)

# Transmission is scaled to give 15,000 e-/s from a mag 10 star
tess_tele_thru = S.UniformTransmission(0.6315)
tess_tele = Telescope(diam=10.5, f_num=1.4, bandpass=tess_tele_thru)

telescope_dict = {'Mono Tele V10UVS (UV Coatings)': mono_tele_v10_vis,
                  'Mono Tele V10UVS (Vis/SWIR Coatings)': mono_tele_v10_vis,
                  'Mono Tele V8UVS (UV Coatings)': mono_tele_v8_uv,
                  'Mono Tele V8UVS (Vis/SWIR Coatings)': mono_tele_v8_vis,
                  'Mono Tele V20UVS (Vis/SWIR Coatings)': mono_tele_v20_vis,
                  'Mono Tele V3UV': mono_tele_v3uv,
                  'Mono Tele V3SWIR': mono_tele_v3swir,
                  'TESS Telescope': tess_tele}

# Defining filters
no_filter = S.UniformTransmission(1)
johnson_u = S.ObsBandpass('johnson,u')
johnson_b = S.ObsBandpass('johnson,b')
johnson_v = S.ObsBandpass('johnson,v')
johnson_r = S.ObsBandpass('johnson,r')
johnson_i = S.ObsBandpass('johnson,i')
johnson_j = S.ObsBandpass('johnson,j')
# Array with uniform total transmission 9000-17000 ang
swir_wave = np.arange(9000, 17000, 100)
swir_thru = np.ones(len(swir_wave))
swir_filt_arr = np.array([swir_wave, swir_thru]).T
# Pad with zeros at 8900 and 17100 ang
swir_filt_arr = np.vstack(([8900, 0], swir_filt_arr, [17100, 0]))
swir_filter = S.ArrayBandpass(swir_filt_arr[:, 0], swir_filt_arr[:, 1])

ultrasat_filt_arr = np.genfromtxt(data_folder + 'ULTRASAT_Filter.csv',
                                  delimiter=',')
ultrasat_filter = S.ArrayBandpass(ultrasat_filt_arr[:, 0],
                                  ultrasat_filt_arr[:, 1] / 100)


filter_dict = {'None': no_filter, 'Johnson U': johnson_u,
               'Johnson B': johnson_b, 'Johnson V': johnson_v,
               'Johnson R': johnson_r, 'Johnson I': johnson_i,
               'Johnson J': johnson_j,
               'ULTRASAT': ultrasat_filter,
               'SWIR (900-1700 nm 100%)': swir_filter}
