# Chris Layden

import os
from Observatory import Sensor, Telescope, Observatory
import pysynphot as S
import numpy as np

data_folder = os.path.dirname(__file__) + '/../data/'

# Sensor dark currents given at -25 deg C, extrapolated from QHY600 data
imx455_qe = S.FileBandpass(data_folder + 'imx455.fits')
imx455 = Sensor(pix_size=3.76, read_noise=1.65, dark_current=1.5*10**-3,
                qe=imx455_qe)

imx487_qe = S.FileBandpass(data_folder + 'imx487.fits')
imx487 = Sensor(pix_size=3.76, read_noise=3, dark_current=5**-4,
                qe=imx487_qe)

v10_bandpass = S.UniformTransmission(0.693)
mono_tele_v10 = Telescope(diam=25, f_num=8, bandpass=v10_bandpass)

v8_bandpass = S.UniformTransmission(0.693)
mono_tele_v8 = Telescope(diam=20, f_num=5, bandpass=v8_bandpass)