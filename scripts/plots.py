'''Makes plots related to TESS-GEO Photometry'''

import numpy as np
import matplotlib.pyplot as plt
import os
import pysynphot as S

data_folder = os.path.dirname(__file__) + '/../data/'
imx571_old_qe = S.FileBandpass(data_folder + 'imx571.fits')
imx487_old_qe = S.FileBandpass(data_folder + 'imx487.fits')

imx571_meas_wave = np.array([350, 400, 450, 500, 550, 600, 640, 700, 750, 800, 830, 850, 880, 905, 940, 980, 1000, 1064])
imx571_meas_qe = np.array([10.52, 55.84, 74.13, 77.82, 71.64, 62.65, 52.06, 37.87, 28.63, 20.31, 16.33, 13.70, 10.50, 8.96, 6.07, 3.58, 2.48, 0.63]) / 100

imx487_meas_wave = np.array([250, 300, 350, 400, 450, 500, 550, 600, 640, 700, 750, 800, 830, 850, 880, 905, 940, 980, 1000, 1064])
imx487_meas_wave_hamamatsu = np.array([250, 300, 350, 400])
imx487_meas_qe = np.array([10.10, 6.65, 38.30, 44.67, 48.92, 51.06, 46.56, 43.80, 38.06, 31.63, 26.79, 18.13, 15.67, 13.80, 11.01, 9.62, 6.69, 4.16, 2.65, 0.64]) / 100
imx487_meas_qe_hamamatsu = np.array([10.10, 6.65, 43.69, 46.57]) / 100

# plt.plot(imx571_old_qe.wave / 10, imx571_old_qe.throughput, label='QHY (Alternate Vendor) Spec')
# plt.scatter(imx571_meas_wave, imx571_meas_qe, label='Measured', c='r')
# plt.legend()
# plt.xlabel('Wavelength (nm)')
# plt.ylabel('Quantum Efficiency')
# plt.ylim([0, 1])
# plt.title('IMX571 Quantum Efficiency')
# plt.show()

plt.plot(imx487_old_qe.wave / 10, imx487_old_qe.throughput, label='Lucid Vision Labs (Vendor) Spec')
plt.scatter(imx487_meas_wave, imx487_meas_qe, label='Measured (ThorLabs PD)', c='r')
plt.scatter(imx487_meas_wave_hamamatsu, imx487_meas_qe_hamamatsu, label='Measured (Hamamatsu PD)', c='g')
plt.legend()
plt.xlabel('Wavelength (nm)')
plt.ylabel('Quantum Efficiency')
plt.ylim([0, 1])
plt.title('IMX487 Quantum Efficiency')
plt.show()
