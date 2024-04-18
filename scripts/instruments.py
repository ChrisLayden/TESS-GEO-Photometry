# Chris Layden

'''Defining sensors and telescopes used in TESS-GEO.'''

import os
import pysynphot as S
import numpy as np
from observatory import Observatory, Sensor, Telescope

data_folder = os.path.dirname(__file__) + '/../data/'

# Defining sensors
# Sensor dark currents given at -25 deg C, extrapolated from QHY600 data
imx455_qe = S.FileBandpass(data_folder + 'imx455.fits')
imx455 = Sensor(pix_size=3.76, read_noise=1.65, dark_current=1.5*10**-3,
                full_well=51000, qe=imx455_qe)


# imx487_qe = S.FileBandpass(data_folder + 'imx487.fits')
imx487_arr = np.genfromtxt(data_folder + 'IMX487_QE_Aaron.csv', delimiter=',')
imx487_qe = S.ArrayBandpass(imx487_arr[:, 0], imx487_arr[:, 1])
imx487 = Sensor(pix_size=2.74, read_noise=2.51, dark_current=5**-4,
                full_well=9662, qe=imx487_qe)

gsense2020_arr = np.genfromtxt(data_folder + 'GSENSE2020_QE.csv', delimiter=',')
# Multiply the first column by 10 to convert from nm to Angstroms
gsense2020_qe = S.ArrayBandpass(gsense2020_arr[:, 0] * 10, gsense2020_arr[:, 1])
# Dark current at -20 C
gsense2020 = Sensor(pix_size=6.5, read_noise=2.67, dark_current=0.2,
                    full_well=54000, qe=gsense2020_qe)
# Dark current at -40 C
gsense4040 = Sensor(pix_size=9, read_noise=2.3, dark_current=0.04,
                    full_well=39000, qe=gsense2020_qe)

ultrasat_arr = np.genfromtxt(data_folder + 'ULTRASAT_FULL_QE.csv', delimiter=',')
# Multiply the first column by 10 to convert from nm to Angstroms
ultrasat_arr[:, 0] *= 10
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

tess_arr = np.genfromtxt(data_folder + 'TESS_throughput.csv', delimiter=',')
tess_throughput = S.ArrayBandpass(tess_arr[:, 0], tess_arr[:, 1])
# This dark current is just a place holder; it's negligible anyways
tesscam = Sensor(pix_size=15, read_noise=10, dark_current=5**-4,
                 full_well=200000, qe=tess_throughput)

basic_sensor = Sensor(pix_size=10, read_noise=10, dark_current=0.01,
                      full_well=100000)

sensor_dict = {'Define New Sensor': basic_sensor,
               'IMX 455 (Visible)': imx455, 'IMX 487 (UV)': imx487,
               'GSENSE2020 (UV)': gsense2020,
               'TESS CCD': tesscam, 'ULTRASAT CMOS': ultrasat_cmos,
               'IMX 990 (SWIR)': imx990}


# Defining telescopes
v10_bandpass = S.UniformTransmission(0.693)
mono_tele_v10 = Telescope(diam=25, f_num=8, psf_type='airy', bandpass=v10_bandpass)

mono_tele_v8_uv = Telescope(diam=17.5, f_num=4.5, psf_type='gaussian',
                            spot_size=2, bandpass=S.UniformTransmission(0.638))
mono_tele_v8_vis = Telescope(diam=17.5, f_num=4.5, psf_type='airy',
                             bandpass=S.UniformTransmission(0.758))

# V10 UVS telescope with UV coatings
mono_tele_v10_uv = Telescope(diam=25, f_num=4.8, psf_type='gaussian',
                             spot_size=2, bandpass=S.UniformTransmission(0.638))
# V10 UVS telescope with visible coatings
mono_tele_v10_vis = Telescope(diam=25, f_num=4.8, psf_type='airy',
                              bandpass=S.UniformTransmission(0.758))


mono_tele_v20_vis = Telescope(diam=47, f_num=4.8, psf_type='airy',
                              bandpass=S.UniformTransmission(0.758))

v3uv_bandpass = S.UniformTransmission(0.54)
v3swir_bandpass = S.UniformTransmission(0.54*0.95/0.8)
mono_tele_v3uv = Telescope(diam=8.5, f_num=3.6, psf_type='gaussian',
                           spot_size=2, bandpass=v3uv_bandpass)
mono_tele_v3swir = Telescope(diam=8.5, f_num=3.6, psf_type='airy', bandpass=v3swir_bandpass)

tess_tele = Telescope(diam=10.5, f_num=1.4, psf_type='gaussian', spot_size=22.91)

basic_tele = Telescope(diam=10, f_num=1)

telescope_dict = {'Define New Telescope': basic_tele, 
                  'Mono Tele V10UVS (UV Coatings)': mono_tele_v10_uv,
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

# Array with uniform total transmission 4000-7000 ang
vis_wave = np.arange(4000, 7000, 100)
vis_thru = np.ones(len(vis_wave))
vis_filt_arr = np.array([vis_wave, vis_thru]).T
# Pad with zeros
vis_filt_arr = np.vstack(([3900, 0], vis_filt_arr, [7100, 0]))
vis_filter = S.ArrayBandpass(vis_filt_arr[:, 0], vis_filt_arr[:, 1])

filter_dict = {'None': no_filter, 'Johnson U': johnson_u,
               'Johnson B': johnson_b, 'Johnson V': johnson_v,
               'Johnson R': johnson_r, 'Johnson I': johnson_i,
               'Johnson J': johnson_j,
               'ULTRASAT': ultrasat_filter,
               'SWIR (900-1700 nm 100%)': swir_filter,
               'Visible (400-700 nm 100%)': vis_filter}

if __name__ == '__main__':
    # Load tess jitter profile
    tess_psd = np.genfromtxt(data_folder + 'TESS_Jitter_PSD.csv', delimiter=',')
    from jitter_tools import integrated_stability
    freqs = np.linspace(1 / 60, 20, 10000)
    amplitudes = 1 / freqs ** 2
    jitter = 1.2
    one_sigma = integrated_stability(1, freqs, amplitudes)
    norm_factor = (jitter / one_sigma) ** 2
    psd = np.array([freqs, norm_factor * amplitudes]).T
    tess_obs = Observatory(telescope=tess_tele, sensor=tesscam, exposure_time=2,
                           num_exposures=1440, jitter_psd=tess_psd)
    vis_obs = Observatory(telescope=mono_tele_v10_vis, sensor=imx455, exposure_time=2,
                          num_exposures=1800, jitter_psd=psd, filter_bandpass=vis_filter)
    uv_obs = Observatory(telescope=mono_tele_v10_uv, sensor=imx487, exposure_time=2,
                          num_exposures=1800, jitter_psd=psd, filter_bandpass=ultrasat_filter)
    spec = S.FlatSpectrum(0.3631, fluxunits='Jy')
    vis_eff_area = vis_obs.eff_area
    uv_eff_area = uv_obs.eff_area
    import matplotlib.pyplot as plt
    plt.plot(vis_eff_area.wave, vis_eff_area.throughput, label='Visible Camera')
    plt.plot(uv_eff_area.wave, uv_eff_area.throughput, label='UV Camera')
    plt.xlabel('Wavelength (Angstroms)')
    plt.ylabel('Effective Area (cm^2)')
    plt.legend()
    plt.show()
    # print(tess_obs.observe(spec, num_frames=300, resolution=11))
    # import time
    # start = time.time()
    # print(vis_obs.observe(spec, num_frames=300, img_size=31, resolution=11))
    # print(time.time() - start)
    # print(vis_obs.observe(spec, aper_method='indiv_aper', num_frames=300, img_size=11))
