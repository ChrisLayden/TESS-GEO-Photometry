'''Makes plots related to TESS-GEO Photometry'''

import os
import numpy as np
import matplotlib.pyplot as plt
import pysynphot as S
from observatory import Observatory
from instruments import sensor_dict, telescope_dict, filter_dict

airy_fwhm = 1.025 * 2500 * 3.5 / 10 ** 4
# Factor of 2 because PSF is ~twice diffraction limited
uv_sigma = 2 * airy_fwhm / 2.355

ultrasat_cmos = sensor_dict['ULTRASAT CMOS']
ultrasat_filter = filter_dict['ULTRASAT']
uv_obs = Observatory(ultrasat_cmos, telescope_dict['Mono Tele V11 (UV)'],
                     exposure_time=300, num_exposures=3, psf_sigma=uv_sigma,
                     filter_bandpass=ultrasat_filter, eclip_lat=90)

uv_mono_tele_bandpass = S.UniformTransmission(0.54)
mono_tele_v11 = telescope_dict['Mono Tele V11 (UV)']
mono_tele_v3 = telescope_dict['Mono Tele V3 (UV)']

imx487 = sensor_dict['IMX 487 (UV)']

data_folder = os.path.dirname(__file__) + '/../data/'
uv_filter = S.FileBandpass(data_folder + "uv_200_300.fits")

tess_geo_v11 = Observatory(imx487, mono_tele_v11,
                           filter_bandpass=uv_filter, exposure_time=60,
                           num_exposures=1, psf_sigma=uv_sigma,
                           eclip_lat=90)

tess_geo_v3 = Observatory(imx487, mono_tele_v3, filter_bandpass=uv_filter,
                          exposure_time=60, num_exposures=4,
                          psf_sigma=uv_sigma, eclip_lat=90)

mag_list = range(5, 26)
obs_list = [tess_geo_v11, tess_geo_v3]
obs_name_list = ["28.5 cm", r"8.5 cm ($\times$4)"]
phot_prec_list = np.zeros((len(obs_list), len(mag_list)))
# psf_sigma_list = [None, 2, 4, 6]
# For UV
psf_sigma_list = [None, uv_sigma, 2, 4, 6]
# psf_sigma_name_list = ["~1.5 (Airy Disk)", "2 (Gaussian)",
#                          "4 (Gaussian)", "6 (Gaussian)"]
# psf_sigma_name_list = ["~1.8 (Airy Disk)", "2 (Gaussian)",
#                          "4 (Gaussian)", "6 (Gaussian)"]
psf_sigma_name_list = ["~0.5 (Airy Disk)", "1 (Gaussian)",
                       "2 (Gaussian)", "4 (Gaussian)", "6 (Gaussian)"]
phot_prec_list = np.zeros((len(psf_sigma_list), len(mag_list)))

# for i, psf_sigma in enumerate(psf_sigma_list):
#     tess_geo_v11.psf_sigma = psf_sigma
#     obs = tess_geo_v11
for i, obs in enumerate(obs_list):

    for j, mag in enumerate(mag_list):
        mag_sp = S.FlatSpectrum(fluxdensity=mag, fluxunits='abmag')
        mag_sp.convert('fnu')
        snr = obs.snr(mag_sp)
        phot_prec = 10 ** 6 / snr
        phot_prec_list[i][j] = phot_prec
    plt.plot(mag_list, phot_prec_list[i])

detection_limit = 10 ** 6 / 5
plt.axhline(y=detection_limit, color='k', linestyle='--')
plt.annotate(r'5-$\sigma$ Detection', [10, detection_limit*1.5])
# plt.legend(psf_sigma_name_list, title=r'PSF $\sigma$ (um)')
plt.legend(obs_name_list, title='Telescope Diamter')
plt.yscale('log')
plt.ylim(10, 10 ** 6)
plt.xlabel('AB Magnitude')
plt.ylabel('Photometric Precision (ppm)')
plt.title("UV-Band Non-Jitter Photometric Precision at 60 sec")
plt.show()
