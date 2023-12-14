'''GUI for calculating photometry for TESS-GEO'''

import os
import tkinter as tk
import pysynphot as S
from spectra import *
from observatory import Sensor, Telescope, Observatory
from instruments import sensor_dict, telescope_dict, filter_dict
from tkinter import messagebox

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
            self.sens_labels[i].grid(row=i+2, column=0, padx=padx, pady=pady)
            self.sens_vars.append(tk.DoubleVar())
            self.sens_boxes.append(tk.Entry(self.root, width=10,
                                            textvariable=self.sens_vars[i]))
            self.sens_boxes[i].grid(row=i+2, column=1, padx=padx, pady=pady)

        self.sens_vars[0].set(5)
        self.sens_vars[1].set(3)
        self.sens_vars[2].set(0.01)
        self.sens_vars[3].set(1)
        self.sens_vars[4].set(100000)
        # If you want to select a default sensor
        self.sens_menu_header = tk.Label(self.root, text='Predefined Sensor',
                                         font=['Arial', 14, 'italic'])
        self.sens_menu_header.grid(row=1, column=0, columnspan=1, padx=padx,
                                   pady=pady)
        self.sens_options = list(sensor_dict.keys())
        self.sens_default = tk.StringVar()
        self.sens_default.set(None)
        self.sens_menu = tk.OptionMenu(self.root, self.sens_default,
                                       *self.sens_options)
        self.sens_menu.grid(row=1, column=1, columnspan=1, padx=padx,
                            pady=pady)
        self.sens_default.trace_add('write', self.set_sens)

        # Defining telescope properties
        self.tele_header = tk.Label(self.root, text='Telescope Properties',
                                    font=['Arial', 16, 'bold'])
        self.tele_header.grid(row=7, column=0, columnspan=2, padx=padx,
                              pady=pady)
        self.tele_labels = []
        tele_label_names = ['Diameter (cm)', 'F/number', 'Bandpass']
        self.tele_boxes = []
        self.tele_vars = []
        for i in range(len(tele_label_names)):
            self.tele_labels.append(tk.Label(self.root,
                                             text=tele_label_names[i]))
            self.tele_labels[i].grid(row=i+9, column=0, padx=padx, pady=pady)
            self.tele_vars.append(tk.DoubleVar())
            self.tele_boxes.append(tk.Entry(self.root, width=10,
                                            textvariable=self.tele_vars[i]))
            self.tele_boxes[i].grid(row=i+9, column=1, padx=padx, pady=pady)

        self.tele_vars[0].set(10)
        self.tele_vars[1].set(10)
        self.tele_vars[2].set(1)

        # If you want to select a default telescope
        self.tele_menu_header = tk.Label(self.root, text='Predefined Telescope',
                                         font=['Arial', 14, 'italic'])
        self.tele_menu_header.grid(row=8, column=0, columnspan=1, padx=padx,
                                   pady=pady)
        self.tele_options = list(telescope_dict.keys())
        self.tele_default = tk.StringVar()
        self.tele_default.set(None)
        self.tele_menu = tk.OptionMenu(self.root, self.tele_default,
                                       *self.tele_options)
        self.tele_menu.grid(row=8, column=1, columnspan=1, padx=padx,
                            pady=pady)
        self.tele_default.trace_add('write', self.set_tele)
        # Defining observing properties
        self.obs_header = tk.Label(self.root, text='Observing Properties',
                                   font=['Arial', 16, 'bold'])
        self.obs_header.grid(row=12, column=0, columnspan=2, padx=padx,
                             pady=pady)

        self.obs_labels = []
        obs_label_names = ['Exposure Time (s)', 'Exposures in Stack',
                           'Limiting SNR', 'Ecliptic Latitude (deg)',
                           'RMS Jitter (arcsec)', 'Jittered Subarray Size (pix)']
        self.obs_boxes = []
        self.obs_vars = []
        for i, value in enumerate(obs_label_names):
            self.obs_labels.append(tk.Label(self.root, text=value))
            self.obs_labels[i].grid(row=i+13, column=0, padx=padx, pady=pady)
            if i == 4:
                self.obs_vars.append(tk.DoubleVar())
            else:
                self.obs_vars.append(tk.IntVar())
            self.obs_boxes.append(tk.Entry(self.root, width=10,
                                           textvariable=self.obs_vars[i]))
            self.obs_boxes[i].grid(row=i+13, column=1, padx=padx, pady=pady)

        self.obs_labels.append(tk.Label(self.root, text='Select Filter'))
        self.obs_vars[0].set(60)
        self.obs_vars[1].set(1)
        self.obs_vars[2].set(5)
        self.obs_vars[3].set(90)
        self.obs_vars[4].set(0.0)
        self.obs_vars[5].set(11)
        self.obs_labels[-1].grid(row=19, column=0, padx=padx, pady=pady)
        self.filter_options = list(filter_dict.keys())
        self.filter_default = tk.StringVar()
        self.filter_default.set('None')
        self.filter_menu = tk.OptionMenu(self.root, self.filter_default,
                                         *self.filter_options)
        self.filter_menu.grid(row=19, column=1, padx=padx, pady=pady)

        # Initializing labels that display results
        self.results_header = tk.Label(self.root, text='General Results',
                                       font=['Arial', 16, 'bold'])
        self.results_header.grid(row=0, column=4, columnspan=1, padx=padx,
                                 pady=pady)

        self.run_button = tk.Button(self.root, fg='green',
                                    text='RUN',
                                    command=self.run_calcs)
        self.run_button.grid(row=0, column=5, columnspan=1, padx=padx,
                             pady=pady)

        self.results_labels = []
        results_label_names = ['Pixel Scale (arcsec/pix)', 'Pivot Wavelength (nm)',
                               'PSF FWHM (um)', 'Central Pixel Ensquared Energy',
                               'Effective Area (cm^2)', 'Limiting AB magnitude',
                               'Saturating AB magnitude']
        self.results_data = []
        for i, name in enumerate(results_label_names):
            self.results_labels.append(tk.Label(self.root, text=name))
            self.results_labels[i].grid(row=i+1, column=4, padx=padx, pady=pady)
            self.results_data.append(tk.Label(self.root, fg='red'))
            self.results_data[i].grid(row=i+1, column=5, padx=padx, pady=pady)

        # Set a spectrum to observe
        self.spectrum_header = tk.Label(self.root, text='Spectrum Observation',
                                        font=['Arial', 16, 'bold'])
        self.spectrum_header.grid(row=0, column=6, columnspan=1, padx=padx,
                                  pady=pady)

        self.run_button = tk.Button(self.root, fg='green', text='RUN',
                                    command=self.run_observation)
        self.run_button.grid(row=0, column=7, columnspan=1, padx=padx,
                             pady=pady)

        self.flat_spec_bool = tk.BooleanVar(value=True)
        self.flat_spec_check = tk.Checkbutton(self.root,
                                              text='Flat spectrum at AB mag',
                                              variable=self.flat_spec_bool)
        self.flat_spec_check.grid(row=1, column=6, padx=padx, pady=pady)
        self.flat_spec_mag = tk.DoubleVar(value=20.0)
        self.flat_spec_entry = tk.Entry(self.root, width=10,
                                        textvariable=self.flat_spec_mag)
        self.flat_spec_entry.grid(row=1, column=7, padx=padx, pady=pady)

        self.bb_spec_bool = tk.BooleanVar()
        self.bb_spec_check = tk.Checkbutton(self.root,
                                            text='Blackbody with Temp (in K)',
                                            variable=self.bb_spec_bool)
        self.bb_spec_check.grid(row=2, column=6, padx=padx, pady=pady)
        self.bb_temp = tk.DoubleVar()
        self.bb_spec_entry_1 = tk.Entry(self.root, width=10,
                                        textvariable=self.bb_temp)
        self.bb_spec_entry_1.grid(row=2, column=7, padx=padx, pady=pady)
        self.bb_dist_label = tk.Label(self.root, text='distance (in Mpc)')
        self.bb_dist_label.grid(row=3, column=6, padx=padx, pady=pady)
        self.bb_distance = tk.DoubleVar()
        self.bb_spec_entry_2 = tk.Entry(self.root, width=10,
                                        textvariable=self.bb_distance)
        self.bb_spec_entry_2.grid(row=3, column=7, padx=padx, pady=pady)
        self.bb_lbol_label = tk.Label(self.root,
                                      text='bolometric luminosity (in erg/s)')
        self.bb_lbol_label.grid(row=4, column=6, padx=padx, pady=pady)
        self.bb_lbol = tk.DoubleVar()
        self.bb_spec_entry_3 = tk.Entry(self.root, width=10,
                                        textvariable=self.bb_lbol)
        self.bb_spec_entry_3.grid(row=4, column=7, padx=padx, pady=pady)

        self.user_spec_bool = tk.BooleanVar()
        self.user_spec_check = tk.Checkbutton(self.root,
                                              text='Spectrum named',
                                              variable=self.user_spec_bool)
        self.user_spec_check.grid(row=5, column=6, padx=padx, pady=pady)
        user_spec_label = tk.Label(self.root,
                                   text='(Spectrum must be in spectra.py)')
        user_spec_label.grid(row=6, column=7, padx=padx)
        self.user_spec_name = tk.StringVar()
        self.user_spec_entry = tk.Entry(self.root, width=20,
                                        textvariable=self.user_spec_name)
        self.user_spec_entry.grid(row=5, column=7, padx=padx, pady=pady)

        self.spec_results_labels = []
        spec_results_label_names = ['Signal (e-)', 'Total Noise (e-)', 'Noise Breakdown', 'SNR',
                                    'Photometric Precision (ppm)', 'Optimal Aperture Size (pix)']
        self.spec_results_data = []
        for i, name in enumerate(spec_results_label_names):
            self.spec_results_labels.append(tk.Label(self.root, text=name))
            self.spec_results_labels[i].grid(row=i+7, column=6, padx=padx, pady=pady)
            self.spec_results_data.append(tk.Label(self.root, fg='red'))
            self.spec_results_data[i].grid(row=i+7, column=7, padx=padx, pady=pady)

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
        if self.tele_default.get() == 'Mono Tele V10UVS (UV Coatings)':
            # ~2 times the diffraction limit for f/4.8 and pivot wavelength 275 nm
            self.psf_sigma = 1.15
        elif self.tele_default.get() == 'Mono Tele V9UVS (UV Coatings)':
            self.psf_sigma = 1.15
        elif self.tele_default.get() == 'TESS Telescope':
            self.psf_sigma = 11
        elif self.tele_default.get() == 'Mono Tele V3UV':
            self.psf_sigma = 0.863
        else:
            self.psf_sigma = None
        self.tele_vars[0].set(self.tele.diam)
        self.tele_vars[1].set(self.tele.f_num)
        self.tele_vars[2].set(self.tele.bandpass)

    def set_obs(self):
        sens_vars = [i.get() for i in self.sens_vars]
        if sens_vars[3] == 'ARRAY':
            sens_vars[3] = self.sens.qe
        else:
            sens_vars[3] = S.UniformTransmission(float(sens_vars[3]))
        sens = Sensor(*sens_vars)
        tele_vars = [i.get() for i in self.tele_vars]
        tele_vars[2] = S.UniformTransmission(tele_vars[2])
        tele = Telescope(*tele_vars)
        exposure_time = self.obs_vars[0].get()
        num_exposures = int(self.obs_vars[1].get())
        limiting_snr = self.obs_vars[2].get()
        eclip_angle = self.obs_vars[3].get()
        filter_bp = filter_dict[self.filter_default.get()]
        jitter = self.obs_vars[4].get()
        observatory = Observatory(sens, tele, exposure_time=exposure_time,
                                  num_exposures=num_exposures,
                                  limiting_s_n=limiting_snr,
                                  filter_bandpass=filter_bp,
                                  psf_sigma=self.psf_sigma,
                                  eclip_lat=eclip_angle,
                                  jitter=jitter)
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
            raise ValueError('No spectrum specified')
        return spectrum

    def run_calcs(self):
        try:
            observatory = self.set_obs()
            limiting_mag = observatory.limiting_mag()
            saturating_mag = observatory.saturating_mag()

            self.results_data[0].config(text=format(observatory.pix_scale, '4.3f'))
            self.results_data[1].config(text=format(observatory.lambda_pivot / 10, '4.1f'))
            self.results_data[2].config(text=format(observatory.psf_fwhm(), '4.3f'))
            self.results_data[3].config(text=format(100 * observatory.central_pix_frac(), '4.1f') + '%')
            self.results_data[4].config(text=format(observatory.eff_area(), '4.2f'))
            self.results_data[5].config(text=format(limiting_mag, '4.3f'))
            self.results_data[6].config(text=format(saturating_mag, '4.3f'))
        except ValueError as inst:
            messagebox.showerror('Value Error', inst)

    def run_observation(self):
        try:
            spectrum = self.set_spectrum()
            observatory = self.set_obs()
            img_size = self.obs_vars[5].get()
            results = observatory.observe(spectrum, img_size=img_size)
            signal = int(results['signal'])
            noise = int(results['tot_noise'])
            snr = signal / noise
            phot_prec = 10 ** 6 / snr
            self.spec_results_data[0].config(text=format(signal, '4d'))
            self.spec_results_data[1].config(text=format(noise, '4d'))
            self.spec_results_data[3].config(text=format(snr, '4.3f'))
            noise_str = ('Shot noise: ' + format(results['shot_noise'], '.2f') +
                         '\nDark noise: ' + format(results['dark_noise'], '.2f') +
                         '\nRead noise: ' + format(results['read_noise'], '.2f') +
                         '\nBackground noise: ' + format(results['bkg_noise'], '.2f') +
                         '\nJitter noise: ' + format(results['jitter_noise'], '.2f'))
            self.spec_results_data[2].config(text=noise_str)
            self.spec_results_data[4].config(text=format(phot_prec, '4.3f'))
            self.spec_results_data[5].config(text=format(results['n_aper'], '2d'))
        except ValueError as inst:
            messagebox.showerror('Value Error', inst)


MyGUI()
