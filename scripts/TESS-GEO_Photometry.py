# Chris Layden

import tkinter as tk
import pysynphot as S
from Observatory import Sensor, Telescope, Observatory, blackbody_spec
from Instruments import imx455, imx487, mono_tele_v10, mono_tele_v8


class MyGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Photometry Calculations")

        padx = 10
        pady = 5

        # Defining sensor properties
        self.sensor_header = tk.Label(self.root, text="Sensor Properties",
                                      font=["Arial", 16, "bold"])
        
        self.sensor_header.grid(row=0, column=0, columnspan=2,
                                padx=padx, pady=pady)

        self.sensor_labels = []
        sensor_label_names = ["Pixel Size (um)", "Read Noise (e-/pix)", 
                              "Dark Current (e-/pix/s)", "Quantum Efficiency"]
        self.sensor_entries = []
        self.sensor_vars = []
        for i in range(len(sensor_label_names)):
            self.sensor_labels.append(tk.Label(self.root, text=sensor_label_names[i]))
            self.sensor_labels[i].grid(row=i+1, column=0, padx=padx, pady=pady)
            self.sensor_vars.append(tk.DoubleVar())
            self.sensor_entries.append(tk.Entry(self.root, width=10,
                                                textvariable=self.sensor_vars[i]))
            self.sensor_entries[i].grid(row=i+1, column=1, padx=padx, pady=pady)

        # If you want to select a default sensor
        self.sensor_menu_header = tk.Label(self.root, text="Or Choose a Sensor",
                                      font=["Arial", 16, "bold"])
        self.sensor_menu_header.grid(row=0, column=2, columnspan=2, padx=padx, pady=pady)
        self.sensor_options = ["IMX 455 (Visible)", "IMX 487 (UV)"]
        self.sensor_default = tk.StringVar()
        self.sensor_default.set(None)
        self.sensor_menu = tk.OptionMenu(self.root, self.sensor_default, *self.sensor_options)
        self.sensor_menu.grid(row=1, column=2, columnspan=2, padx=padx, pady=pady)
        self.sensor_default.trace("w", self.set_sensor)

        # Defining telescope properties
        self.telescope_header = tk.Label(self.root, text="Telescope Properties",
                                      font=["Arial", 16, "bold"])
        self.telescope_header.grid(row=6, column=0, columnspan=2, padx=padx, pady=pady)
        self.telescope_labels = []
        telescope_label_names = ["Diameter (cm)", "F/number",
                                 "Bandpass"]
        self.telescope_entries = []
        self.telescope_vars = []
        for i in range(len(telescope_label_names)):
            self.telescope_labels.append(tk.Label(self.root, text=telescope_label_names[i]))
            self.telescope_labels[i].grid(row=i+7, column=0, padx=padx, pady=pady)
            self.telescope_vars.append(tk.DoubleVar())
            self.telescope_entries.append(tk.Entry(self.root, width=10,
                                          textvariable=self.telescope_vars[i]))
            self.telescope_entries[i].grid(row=i+7, column=1, padx=padx, pady=pady)

         # If you want to select a default telescope
        self.telescope_menu_header = tk.Label(self.root, text="Or Choose a Telescope",
                                      font=["Arial", 16, "bold"])
        self.telescope_menu_header.grid(row=6, column=2, columnspan=2, padx=padx, pady=pady)
        self.telescope_options = ["MonoTele V10 (Visible)", "MonoTele V8 (UV)"]
        self.telescope_default = tk.StringVar()
        self.telescope_default.set(None)
        self.telescope_menu = tk.OptionMenu(self.root, self.telescope_default, *self.telescope_options)
        self.telescope_menu.grid(row=7, column=2, columnspan=2, padx=padx, pady=pady)
        self.telescope_default.trace("w", self.set_telescope)

        # Defining observing properties
        self.observing_header = tk.Label(self.root, text="Observing Properties",
                                      font=["Arial", 16, "bold"])
        self.observing_header.grid(row=10, column=0, columnspan=2, padx=padx, pady=pady)         

        self.observing_labels = []
        observing_label_names = ["Exposure Time (s)", "Exposures in Stack",
                                 "Limiting SNR"]
        self.observing_entries = []
        self.observing_vars = []
        for i in range(len(observing_label_names)):
            self.observing_labels.append(tk.Label(self.root, text=observing_label_names[i]))
            self.observing_labels[i].grid(row=i+11, column=0, padx=padx, pady=pady)
            self.observing_vars.append(tk.DoubleVar())
            self.observing_entries.append(tk.Entry(self.root, width=10,
                                          textvariable=self.observing_vars[i]))
            self.observing_entries[i].grid(row=i+11, column=1, padx=padx, pady=pady)

        self.observing_labels.append(tk.Label(self.root, text="Select Filter"))
        self.observing_labels[-1].grid(row=14, column=0, padx=padx, pady=pady)
        self.filter_options = ["None", "Johnson U", "Johnson V",
                               "Johnson B", "Johnson R", "Johnson I"]
        self.filter_default = tk.StringVar()
        self.filter_default.set("None")
        self.filter_menu = tk.OptionMenu(self.root, self.filter_default, *self.filter_options)
        self.filter_menu.grid(row=14, column=1, padx=padx, pady=pady)

        self.lim_mag_button = tk.Button(self.root, text="Calculate limiting magnitude", command=self.limiting_mag)
        self.lim_mag_button.grid(row=15, column=0, columnspan=2, padx=padx, pady=pady)

        # Set a spectrum to observe
        self.spectrum_header = tk.Label(self.root, text="Spectrum to Observe",
                                      font=["Arial", 16, "bold"])
        self.spectrum_header.grid(row=0, column=4, columnspan=2, padx=padx, pady=pady)

        self.flat_spec_bool = tk.BooleanVar()
        self.flat_spec_check = tk.Checkbutton(self.root, text="Flat spectrum at AB magnitude",
                                              variable=self.flat_spec_bool)
        self.flat_spec_check.grid(row=1, column=4, padx=padx, pady=pady)
        self.flat_spec_mag = tk.DoubleVar()
        self.flat_spec_entry = tk.Entry(self.root, width=10, textvariable=self.flat_spec_mag)
        self.flat_spec_entry.grid(row=1, column=5, padx=padx, pady=pady)

        self.bb_spec_bool = tk.BooleanVar()
        self.bb_spec_check = tk.Checkbutton(self.root, text="Blackbody with temperature (in K)",
                                            variable=self.bb_spec_bool)
        self.bb_spec_check.grid(row=2, column=4, padx=padx, pady=pady)
        self.bb_temp = tk.DoubleVar()
        self.bb_spec_entry_1 = tk.Entry(self.root, width=10, textvariable=self.bb_temp)
        self.bb_spec_entry_1.grid(row=2, column=5, padx=padx, pady=pady)
        tk.Label(self.root, text="distance (in Mpc)").grid(row=3, column=4, padx=padx, pady=pady)
        self.bb_distance = tk.DoubleVar()
        self.bb_spec_entry_2 = tk.Entry(self.root, width=10, textvariable=self.bb_distance)
        self.bb_spec_entry_2.grid(row=3, column=5, padx=padx, pady=pady)
        tk.Label(self.root, text="bolometric luminosity (in erg/s)").grid(row=4, column=4, padx=padx, pady=pady)
        self.bb_lbol = tk.DoubleVar()
        self.bb_spec_entry_3 = tk.Entry(self.root, width=10, textvariable=self.bb_lbol)
        self.bb_spec_entry_3.grid(row=4, column=5, padx=padx, pady=pady)

        self.user_spec_bool = tk.BooleanVar()
        self.user_spec_check = tk.Checkbutton(self.root, text="User-defined spectrum with file name",
                                              variable=self.user_spec_bool)
        self.user_spec_check.grid(row=5, column=4, padx=padx, pady=pady)
        user_spec_label = tk.Label(self.root, text="(specify wavelength in Angstroms \nand flux in erg/s/cm^2/Hz)")
        user_spec_label.grid(row=6, column=4, padx=padx)
        self.user_spec_name = tk.StringVar()
        self.user_spec_name.set("Not implemented yet")
        self.user_spec_entry = tk.Entry(self.root, width=20, textvariable=self.user_spec_name)
        self.user_spec_entry.grid(row=5, column=5, padx=padx, pady=pady)

        self.signal_button = tk.Button(self.root, text="Calculate Signal-to-Noise Ratio", command=self.calc_signal)
        self.signal_button.grid(row=8, column=4, columnspan=2, padx=padx, pady=pady)

        self.root.mainloop()

    def set_sensor(self, *args):
        if self.sensor_default.get() == self.sensor_options[0]:
            self.sensor = imx455
        elif self.sensor_default.get() == self.sensor_options[1]:
            self.sensor = imx487
        self.sensor_vars[0].set(self.sensor.pix_size)
        self.sensor_vars[1].set(self.sensor.read_noise)
        self.sensor_vars[2].set(self.sensor.dark_current)
        self.sensor_entries[3].delete(0,10)
        self.sensor_entries[3].insert(0,"ARRAY")

    def set_telescope(self, *args):
        if self.telescope_default.get() == self.telescope_options[0]:
            self.telescope = mono_tele_v10
        elif self.telescope_default.get() == self.telescope_options[1]:
            self.telescope = mono_tele_v8
        self.telescope_vars[0].set(self.telescope.diam)
        self.telescope_vars[1].set(self.telescope.f_num)
        self.telescope_vars[2].set(self.telescope.bandpass)

    def set_obs(self):
        # self.sensor_entries[3].config({"background": "Black"})
        try:
            sensor_vars = [i.get() for i in self.sensor_vars]
        except tk.TclError:
            sensor_vars = [self.sensor_vars[0].get(), self.sensor_vars[1].get(),
                           self.sensor_vars[2].get(), self.sensor.qe]
        sensor = Sensor(*sensor_vars)
        telescope_vars = [i.get() for i in self.telescope_vars]
        telescope = Telescope(*telescope_vars)
        exposure_time = self.observing_vars[0].get()
        num_exposures = self.observing_vars[1].get()
        limiting_snr = self.observing_vars[2].get()
        if self.filter_default.get() == "None":
            filter_bp = 1
        else:
            filter_name = self.filter_default.get()
            filter_str = "johnson," + filter_name[-1].lower()
            filter_bp = S.ObsBandpass(filter_str)
        observatory = Observatory(sensor, telescope, exposure_time=exposure_time,
                                  num_exposures=num_exposures, limiting_s_n=limiting_snr,
                                  filter_bandpass=filter_bp)
        return observatory

    def limiting_mag(self):
        observatory = self.set_obs()
        flat_spec = S.FlatSpectrum(15, fluxunits='abmag')
        flat_spec.convert('fnu')
        try:
            limiting_mag = observatory.limiting_mag()
            self.lim_mag_label = tk.Label(self.root, text="Limiting Magnitude: " + 
                                        format(limiting_mag,'4.3f'), fg='red', bg='white')
            self.lim_mag_label.grid(row=16, column=0, columnspan=2, padx=10, pady=5)
            return limiting_mag
        except AttributeError:
            print("ERROR: At least one of QE, telescope bandpass, or filter bandpass must be array-like.")

    def calc_signal(self):
        if self.flat_spec_bool.get():
            spectrum = S.FlatSpectrum(fluxdensity=self.flat_spec_mag.get(), fluxunits='abmag')
            spectrum.convert('fnu')
        elif self.bb_spec_bool.get():
            temp = self.bb_temp.get()
            distance = self.bb_distance.get()
            l_bol = self.bb_lbol.get()
            spectrum = blackbody_spec(temp, distance, l_bol)
            
        elif self.user_spec_bool.get():
            print("Not functional yet")
        else:
            raise("No spectrum specified")
        
        observatory = self.set_obs()
        snr, ap = observatory.snr(spectrum)
        self.snr_label = tk.Label(self.root, text="Observed SNR: " + 
                                        format(snr,'4.3f'), fg='red', bg='white')
        self.snr_label.grid(row=9, column=4, columnspan=2, padx=10, pady=5)

MyGUI()