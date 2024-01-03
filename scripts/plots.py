'''Makes plots related to TESS-GEO Photometry'''

import os
import numpy as np
import matplotlib.pyplot as plt
import pysynphot as S
from astropy.io import fits

data_folder = os.path.dirname(__file__) + '/../data/'
imx571_old_qe = S.FileBandpass(data_folder + 'imx571.fits')
imx487_old_qe = S.FileBandpass(data_folder + 'imx487.fits')

imx571_meas_wave = np.array([350, 400, 450, 500, 550, 600, 640, 700, 750, 800, 830, 850, 880, 905, 940, 980, 1000, 1064])
imx571_meas_qe = np.array([10.52, 55.84, 74.13, 77.82, 71.64, 62.65, 52.06, 37.87, 28.63, 20.31, 16.33, 13.70, 10.50, 8.96, 6.07, 3.58, 2.48, 0.63]) / 100

imx487_meas_wave = np.array([400, 450, 500, 550, 600, 640, 700, 750, 800, 830, 850, 880, 905, 940, 980, 1000, 1064])
imx487_meas_wave_hamamatsu = np.array([250, 300, 350, 400])
imx487_meas_qe = np.array([44.7, 48.9, 51.1, 46.6, 43.8, 38.1, 31.6, 26.8, 18.11, 15.7, 13.8, 11.0, 9.6, 6.7, 4.2, 2.7, 0.64]) / 100
imx487_sigma = np.array([0.013, 0.014, 0.014, 0.012, 0.012, 0.010, 0.009, 0.007, 0.005, 0.004, 0.004, 0.003, 0.003, 0.002, 0.001, 0.001, 0.0002])
imx487_meas_qe_hamamatsu = np.array([10.10, 6.65, 43.69, 46.57]) / 100

# Return a csv file with imx487_meas_wave and imx487_meas_qe, with 3 decimal places
np.savetxt('imx487.csv', np.transpose([imx487_meas_wave, imx487_meas_qe]), delimiter=',', header='Wavelength (nm), QE', fmt='%.3f')


# plt.plot(imx571_old_qe.wave / 10, imx571_old_qe.throughput, label='QHY (Alternate Vendor) Spec')
# plt.scatter(imx571_meas_wave, imx571_meas_qe, label='Measured', c='r')
# plt.legend()
# plt.xlabel('Wavelength (nm)')
# plt.ylabel('Quantum Efficiency')
# plt.ylim([0, 1])
# plt.title('IMX571 Quantum Efficiency')
# plt.show()


plt.plot(imx487_old_qe.wave / 10, imx487_old_qe.throughput, label='Lucid Vision Labs (Vendor) Spec')
# plt.errorbar(imx487_meas_wave, imx487_meas_qe, imx487_sigma, fmt='r.', markersize=5)
plt.scatter(imx487_meas_wave, imx487_meas_qe, label='Measured (ThorLabs PD)', c='r')
# plt.scatter(imx487_meas_wave_hamamatsu, imx487_meas_qe_hamamatsu, label='Measured (Hamamatsu PD)', c='g')
plt.legend()
plt.xlabel('Wavelength (nm)')
plt.ylabel('Quantum Efficiency')
plt.ylim([0, 1])
plt.title('IMX487 Quantum Efficiency')
plt.show()

# wavelengths = [200, 250, 300, 350, 400, 450, 500] # nm
# hamamatsu_pd_response = np.array([0.1329, 0.1226, 0.1273, 0.1477, 0.1801, 0.2197, 0.2521]) # A/W
# hamamatsu_pd_reading = np.array([0.002, 30.9, 25.0, 0.073, 0.232, 0.552, 0.715]) # nA
# hamamatsu_power = hamamatsu_pd_reading / hamamatsu_pd_response # nW
# plt.scatter(wavelengths, hamamatsu_power)
# plt.xlabel('Wavelength (nm)')
# plt.ylabel('Power Read by PD (nW)')
# plt.yscale('log')
# plt.show()
print(np.logspace(1, 100))