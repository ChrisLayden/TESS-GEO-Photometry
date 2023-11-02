import os
import numpy as np
import matplotlib.pyplot as plt

data_folder = os.path.dirname(__file__) + '/../data/'

# Get data for absorption lengths in silicon, SiO2, and polysilicon.
hard_xray_absorption = np.genfromtxt(data_folder + 'xray_absorption.csv', delimiter=',', skip_header=1) # x: keV, y: um
# Get rid of energies below 1.47 keV, where we have soft x-ray data
hard_xray_absorption = hard_xray_absorption[hard_xray_absorption[:, 0] >= 1.47]
soft_xray_absorption = np.genfromtxt(data_folder + 'soft_xray_absorption_si.csv', delimiter=',', skip_header=1) # x: eV, y: um
soft_xray_absorption[:, 0] /= 1000 # Convert eV to keV
# Join the xray absorption arrays
xray_absorption_si = np.concatenate((soft_xray_absorption, hard_xray_absorption))
# Append some more points from NIST
density_si = 2.3290 # g / cm^3
point1 = [15, 10 ** 4 / (10.34 * density_si)]
point2 = [20, 10 ** 4 / (4.464 * density_si)]
point3 = [30, 10 ** 4 / (1.436 * density_si)]
xray_absorption_si = np.append(xray_absorption_si, [point1, point2, point3], axis=0)

# plt.plot(xray_absorption_si[:, 0], xray_absorption_si[:, 1])
# plt.show()

xray_absorption_sio2 = np.genfromtxt(data_folder + 'soft_xray_absorption_sio2.csv', delimiter=',', skip_header=1) # x: eV, y: um
xray_absorption_sio2[:, 0] /= 1000 # Convert eV to keV

fsi_poly_thickness = 0.5 # um
fsi_sio2_thickness = 0.2 # um
bsi_poly_thickness = 0.0 # um
bsi_sio2_thickness = 25 / 10 ** 4 # um # Native growth on the thinned surface
ccid80_dep_depth = 100 # um
ccid20_dep_depth = 30 # um

sensor_area = 2 * (2048 * 4096) * (15 * 15) / 10 ** 8 # cm^2 # 2 CCDs in one SXC

xray_energies = np.linspace(0.2, 25, 1000)
ccid20_qe = np.zeros_like(xray_energies)
ccid80_qe = np.zeros_like(xray_energies)
for i, energy in enumerate(xray_energies):
    fsi_poly_trans = np.exp(-fsi_poly_thickness / np.interp(energy, xray_absorption_si[:,0], xray_absorption_si[:,1]))
    bsi_poly_trans = np.exp(-bsi_poly_thickness / np.interp(energy, xray_absorption_si[:,0], xray_absorption_si[:,1]))
    if energy > xray_absorption_sio2[:,0].max():
        fsi_sio2_trans = 1
        bsi_sio2_trans = 1
    else:
        fsi_sio2_trans = np.exp(-fsi_sio2_thickness / np.interp(energy, xray_absorption_sio2[:,0], xray_absorption_sio2[:,1]))
        bsi_sio2_trans = np.exp(-bsi_sio2_thickness / np.interp(energy, xray_absorption_sio2[:,0], xray_absorption_sio2[:,1]))
    ccid20_abs = 1 - np.exp(-ccid20_dep_depth / np.interp(energy, xray_absorption_si[:,0], xray_absorption_si[:,1]))
    ccid80_abs = 1 - np.exp(-ccid80_dep_depth / np.interp(energy, xray_absorption_si[:,0], xray_absorption_si[:,1]))
    ccid20_qe[i] = fsi_poly_trans * fsi_sio2_trans * ccid20_abs
    ccid80_qe[i] = bsi_poly_trans * bsi_sio2_trans * ccid80_abs

# Plot energy vs. quantum efficiency
plt.plot(xray_energies, ccid20_qe, label='CCID-20')
plt.plot(xray_energies, ccid80_qe, label='CCID-80')
plt.xlabel('Energy (keV)')
plt.ylabel('Quantum Efficiency')
plt.legend(loc='upper right')
plt.show()

# Plot energy vs. effective area
open_fractions = [0.5, 0.35, 0.2]
for open_frac in open_fractions:
    plt.plot(xray_energies, ccid80_qe * sensor_area * open_frac, label='CCID-80, ' + format(100 * open_frac, '2.0f') + '% Open')
plt.plot(xray_energies, ccid20_qe * sensor_area * 0.2, label='CCID-20, 20% Open')
plt.xlabel('Energy (keV)')
plt.ylabel('Effective Area per SXC (cm$^2$)')
plt.legend(loc='upper right')
plt.show()