'''GUI for calculating photometry for TESS-GEO'''

import os
import tkinter as tk
import pysynphot as S
import numpy as np
from spectra import *
from observatory import Sensor, Telescope, Observatory, blackbody_spec
from instruments import sensor_dict, telescope_dict, filter_dict

data_folder = os.path.dirname(__file__) + '/../data/'


class MyGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title('Photometry Calculations')
        self.psf_sigma = None

        padx = 10
        pady = 5

        # Defining sensor properties
        self.sens_header = tk.Label(self.root, text='Sensor Properties',
                                    font=['Arial', 16, 'bold'])

        self.sens_header.grid(row=0, column=0, columnspan=2,
                              padx=padx, pady=pady)

        self.sens_labels = []
        sens_label_names = ['Pixel Size (um)', 'Read Noise (e-/pix)',
                            'Dark Current (e-/pix/s)', 'Quantum Efficiency',
                            'Full Well Capacity']
        self.sens_boxes = []
        self.sens_vars = []
        for i in range(len(sens_label_names)):
            self.sens_labels.append(tk.Label(self.root,
                                             text=sens_label_names[i]))
            self.sens_labels[i].grid(row=i+1, column=0, padx=padx, pady=pady)
            self.sens_vars.append(tk.DoubleVar())
            self.sens_boxes.append(tk.Entry(self.root, width=10,
                                            textvariable=self.sens_vars[i]))
            self.sens_boxes[i].grid(row=i+1, column=1, padx=padx, pady=pady)

        self.sens_vars[0].set(5)
        self.sens_vars[1].set(3)
        self.sens_vars[2].set(0.01)
        self.sens_vars[3].set(1)
        self.sens_vars[4].set(100000)
        # If you want to select a default sensor
        self.sens_menu_header = tk.Label(self.root, text='Or Choose Sensor',
                                         font=['Arial', 16, 'bold'])
        self.sens_menu_header.grid(row=0, column=2, columnspan=2, padx=padx,
                                   pady=pady)
        self.sens_options = list(sensor_dict.keys())
        self.sens_default = tk.StringVar()
        self.sens_default.set(None)
        self.sens_menu = tk.OptionMenu(self.root, self.sens_default,
                                       *self.sens_options)
        self.sens_menu.grid(row=1, column=2, columnspan=2, padx=padx,
                            pady=pady)
        self.sens_default.trace('w', self.set_sens)

        # Defining telescope properties
        self.tele_header = tk.Label(self.root, text='Telescope Properties',
                                    font=['Arial', 16, 'bold'])
        self.tele_header.grid(row=6, column=0, columnspan=2, padx=padx,
                              pady=pady)
        self.tele_labels = []
        tele_label_names = ['Diameter (cm)', 'F/number', 'Bandpass']
        self.tele_boxes = []
        self.tele_vars = []
        for i in range(len(tele_label_names)):
            self.tele_labels.append(tk.Label(self.root,
                                             text=tele_label_names[i]))
            self.tele_labels[i].grid(row=i+7, column=0, padx=padx, pady=pady)
            self.tele_vars.append(tk.DoubleVar())
            self.tele_boxes.append(tk.Entry(self.root, width=10,
                                            textvariable=self.tele_vars[i]))
            self.tele_boxes[i].grid(row=i+7, column=1, padx=padx, pady=pady)

        self.tele_vars[0].set(10)
        self.tele_vars[1].set(10)
        self.tele_vars[2].set(1)

        # If you want to select a default telescope
        self.tele_menu_header = tk.Label(self.root, text='Or Choose Telescope',
                                         font=['Arial', 16, 'bold'])
        self.tele_menu_header.grid(row=6, column=2, columnspan=2, padx=padx,
                                   pady=pady)
        self.tele_options = list(telescope_dict.keys())
        self.tele_default = tk.StringVar()
        self.tele_default.set(None)
        self.tele_menu = tk.OptionMenu(self.root, self.tele_default,
                                       *self.tele_options)
        self.tele_menu.grid(row=7, column=2, columnspan=2, padx=padx,
                            pady=pady)
        self.tele_default.trace('w', self.set_tele)

        # Defining observing properties
        self.obs_header = tk.Label(self.root, text='Observing Properties',
                                   font=['Arial', 16, 'bold'])
        self.obs_header.grid(row=10, column=0, columnspan=2, padx=padx,
                             pady=pady)

        self.obs_labels = []
        obs_label_names = ['Exposure Time (s)', 'Exposures in Stack',
                           'Limiting SNR', 'Ecliptic Latitude (deg)']
        self.obs_boxes = []
        self.obs_vars = []
        for i, value in enumerate(obs_label_names):
            self.obs_labels.append(tk.Label(self.root, text=value))
            self.obs_labels[i].grid(row=i+11, column=0, padx=padx, pady=pady)
            self.obs_vars.append(tk.DoubleVar())
            self.obs_boxes.append(tk.Entry(self.root, width=10,
                                           textvariable=self.obs_vars[i]))
            self.obs_boxes[i].grid(row=i+11, column=1, padx=padx, pady=pady)

        self.obs_labels.append(tk.Label(self.root, text='Select Filter'))
        self.obs_vars[0].set(300)
        self.obs_vars[1].set(3)
        self.obs_vars[2].set(5)
        self.obs_vars[3].set(90)
        self.obs_labels[-1].grid(row=15, column=0, padx=padx, pady=pady)
        self.filter_options = list(filter_dict.keys())
        self.filter_default = tk.StringVar()
        self.filter_default.set('None')
        self.filter_menu = tk.OptionMenu(self.root, self.filter_default,
                                         *self.filter_options)
        self.filter_menu.grid(row=15, column=1, padx=padx, pady=pady)

        # Set a spectrum to observe
        self.spectrum_header = tk.Label(self.root, text='Spectrum to Observe',
                                        font=['Arial', 16, 'bold'])
        self.spectrum_header.grid(row=16, column=0, columnspan=2, padx=padx,
                                  pady=pady)

        self.flat_spec_bool = tk.BooleanVar(value=True)
        self.flat_spec_check = tk.Checkbutton(self.root,
                                              text='Flat spectrum at AB mag',
                                              variable=self.flat_spec_bool)
        self.flat_spec_check.grid(row=17, column=0, padx=padx, pady=pady)
        self.flat_spec_mag = tk.DoubleVar(value=20.0)
        self.flat_spec_entry = tk.Entry(self.root, width=10,
                                        textvariable=self.flat_spec_mag)
        self.flat_spec_entry.grid(row=17, column=1, padx=padx, pady=pady)

        self.bb_spec_bool = tk.BooleanVar()
        self.bb_spec_check = tk.Checkbutton(self.root,
                                            text='Blackbody with Temp (in K)',
                                            variable=self.bb_spec_bool)
        self.bb_spec_check.grid(row=18, column=0, padx=padx, pady=pady)
        self.bb_temp = tk.DoubleVar()
        self.bb_spec_entry_1 = tk.Entry(self.root, width=10,
                                        textvariable=self.bb_temp)
        self.bb_spec_entry_1.grid(row=18, column=1, padx=padx, pady=pady)
        self.bb_dist_label = tk.Label(self.root, text='distance (in Mpc)')
        self.bb_dist_label.grid(row=19, column=0, padx=padx, pady=pady)
        self.bb_distance = tk.DoubleVar()
        self.bb_spec_entry_2 = tk.Entry(self.root, width=10,
                                        textvariable=self.bb_distance)
        self.bb_spec_entry_2.grid(row=19, column=1, padx=padx, pady=pady)
        self.bb_lbol_label = tk.Label(self.root,
                                      text='bolometric luminosity (in erg/s)')
        self.bb_lbol_label.grid(row=20, column=0, padx=padx, pady=pady)
        self.bb_lbol = tk.DoubleVar()
        self.bb_spec_entry_3 = tk.Entry(self.root, width=10,
                                        textvariable=self.bb_lbol)
        self.bb_spec_entry_3.grid(row=20, column=1, padx=padx, pady=pady)

        self.user_spec_bool = tk.BooleanVar()
        self.user_spec_check = tk.Checkbutton(self.root,
                                              text='Spectrum named',
                                              variable=self.user_spec_bool)
        self.user_spec_check.grid(row=21, column=0, padx=padx, pady=pady)
        user_spec_label = tk.Label(self.root,
                                   text='(Spectrum must be in spectra.py)')
        user_spec_label.grid(row=22, column=1, padx=padx)
        self.user_spec_name = tk.StringVar()
        self.user_spec_entry = tk.Entry(self.root, width=20,
                                        textvariable=self.user_spec_name)
        self.user_spec_entry.grid(row=21, column=1, padx=padx, pady=pady)

        # Initializing labels that display results
        self.results_header = tk.Label(self.root, text='Tabulated Results',
                                        font=['Arial', 16, 'bold'])
        self.results_header.grid(row=0, column=4, columnspan=2, padx=padx,
                                  pady=pady)

        self.run_button = tk.Button(self.root, fg='green',
                                        text='RUN',
                                        command=self.run_calcs)
        self.run_button.grid(row=1, column=4, columnspan=2, padx=padx,
                                 pady=pady)
        
        self.pix_scale_button = tk.Label(self.root, text='Pixel Scale (arcsec/pix)')
        self.pix_scale_button.grid(row=2, column=4, columnspan=1, padx=padx,
                                 pady=pady)

        self.lim_mag_button = tk.Label(self.root,
                                        text='Limiting AB magnitude')
        self.lim_mag_button.grid(row=3, column=4, columnspan=1, padx=padx,
                                 pady=pady)

        self.sat_mag_button = tk.Label(self.root,
                                        text='Saturating AB magnitude')
        self.sat_mag_button.grid(row=4, column=4, columnspan=1, padx=padx,
                                 pady=pady)

        self.signal_button = tk.Label(self.root, text='Signal (e-)')
        self.signal_button.grid(row=5, column=4, columnspan=1, padx=padx,
                                pady=pady)

        self.snr_button = tk.Label(self.root, text='SNR')
        self.snr_button.grid(row=6, column=4, columnspan=1, padx=padx,
                             pady=pady)

        self.phot_prec_button = tk.Label(self.root, fg='black',
                                          text='Photometric Precision (ppm)')
        self.phot_prec_button.grid(row=7, column=4, columnspan=1, padx=padx,
                                   pady=pady)

        self.n_aper_button = tk.Label(self.root, fg='black',
                                          text='Pixels in Optimal Aperture')
        self.n_aper_button.grid(row=8, column=4, columnspan=1, padx=padx,
                                   pady=pady)

        self.pix_scale_label = tk.Label(self.root, fg='red')
        self.pix_scale_label.grid(row=2, column=5, columnspan=2, padx=10,
                                  pady=5)
        self.lim_mag_label = tk.Label(self.root, fg='red')
        self.lim_mag_label.grid(row=3, column=5, columnspan=2, padx=10,
                                pady=5)
        self.sat_mag_label = tk.Label(self.root, fg='red')
        self.sat_mag_label.grid(row=4, column=5, columnspan=2, padx=10,
                                pady=5)
        self.sig_label = tk.Label(self.root, fg='red')
        self.sig_label.grid(row=5, column=5, columnspan=1, padx=10, pady=5)
        self.snr_label = tk.Label(self.root, fg='red')
        self.snr_label.grid(row=6, column=5, columnspan=1, padx=10, pady=5)
        self.phot_prec_label = tk.Label(self.root, fg='red')
        self.phot_prec_label.grid(row=7, column=5, columnspan=1,
                                  padx=10, pady=5)
        self.n_aper_label = tk.Label(self.root, fg='red')
        self.n_aper_label.grid(row=8, column=5, columnspan=1,
                                  padx=10, pady=5)

        self.root.mainloop()

    def set_sens(self, *args):
        self.sens = sensor_dict[self.sens_default.get()]
        self.sens_vars[0].set(self.sens.pix_size)
        self.sens_vars[1].set(self.sens.read_noise)
        self.sens_vars[2].set(self.sens.dark_current)
        self.sens_vars[4].set(self.sens.full_well)
        self.sens_vars[3] = tk.StringVar()
        self.sens_boxes[3].config(textvariable=self.sens_vars[3])
        self.sens_vars[3].set('ARRAY')

    def set_tele(self, *args):
        self.tele = telescope_dict[self.tele_default.get()]
        if self.tele_default.get() == 'Mono Tele V11 (UV)':
            # ~2 times the diffraction limit for f/3.5
            self.psf_sigma = 0.76
        self.tele_vars[0].set(self.tele.diam)
        self.tele_vars[1].set(self.tele.f_num)
        self.tele_vars[2].set(self.tele.bandpass)

    def set_obs(self):
        # try:
        sens_vars = [i.get() for i in self.sens_vars]
        if sens_vars[3] == 'ARRAY':
            sens_vars[3] = self.sens.qe
        else:
            sens_vars[3] = S.UniformTransmission(float(sens_vars[3]))
        sens = Sensor(*sens_vars)
        # except tk.TclError:
        #     sens = self.sens
        tele_vars = [i.get() for i in self.tele_vars]
        tele_vars[2] = S.UniformTransmission(tele_vars[2])
        tele = Telescope(*tele_vars)
        exposure_time = self.obs_vars[0].get()
        num_exposures = self.obs_vars[1].get()
        limiting_snr = self.obs_vars[2].get()
        eclip_angle = self.obs_vars[3].get()
        filter_bp = filter_dict[self.filter_default.get()]
        observatory = Observatory(sens, tele, exposure_time=exposure_time,
                                  num_exposures=num_exposures,
                                  limiting_s_n=limiting_snr,
                                  filter_bandpass=filter_bp,
                                  psf_sigma=self.psf_sigma,
                                  eclip_lat=eclip_angle)
        return observatory

    def set_spectrum(self):
        if self.flat_spec_bool.get():
            spectrum = S.FlatSpectrum(fluxdensity=self.flat_spec_mag.get(),
                                      fluxunits='abmag')
            spectrum.convert('fnu')
        elif self.bb_spec_bool.get():
            temp = self.bb_temp.get()
            distance = self.bb_distance.get()
            l_bol = self.bb_lbol.get()
            spectrum = blackbody_spec(temp, distance, l_bol)
        elif self.user_spec_bool.get():
            spectrum_name = self.user_spec_name.get()
            spectrum = eval(spectrum_name)
        else:
            raise 'No spectrum specified'
        return spectrum


    def run_calcs(self):
        spectrum = self.set_spectrum()
        observatory = self.set_obs()
        (signal, noise, obs_grid, aper) = observatory.observation(spectrum)
        signal = round(signal)
        snr = signal / noise
        phot_prec = 10 ** 6 / snr
        n_aper = int(np.sum(aper))
        limiting_mag = observatory.limiting_mag()
        saturating_mag = observatory.saturating_mag()

        self.pix_scale_label.config(text=format(observatory.pix_scale, '4.3f'))
        self.lim_mag_label.config(text=format(limiting_mag, '4.3f'))
        self.sat_mag_label.config(text=format(saturating_mag, '4.3f'))
        self.sig_label.config(text=format(signal, '4d'))
        self.snr_label.config(text=format(snr, '4.3f'))
        self.phot_prec_label.config(text=format(phot_prec, '4.3f'))
        self.n_aper_label.config(text=format(n_aper, '2d'))

MyGUI()
