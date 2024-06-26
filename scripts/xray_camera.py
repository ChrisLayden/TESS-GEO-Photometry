# Given layers of a certain thickness/composition and device depletion depth,
# calculate the quantum efficiency of a CCD/CMOS sensor for a given energy range.

import os
import copy
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize

# Set text size in all matplotlib plots
plt.rcParams.update({'font.size': 14})

data_folder = os.path.dirname(__file__) + '/../data/'

# Get data for absorption lengths in silicon, SiO2, polyimide, ZZZ.
SiO2_absorption = np.genfromtxt(data_folder + 'SiO2_absorption.csv', delimiter=',', skip_header=1) # x: eV, y: um
Si_absorption = np.genfromtxt(data_folder + 'Si_absorption.csv', delimiter=',', skip_header=1) # x: eV, y: um
Si3N4_absorption = np.genfromtxt(data_folder + 'Si3N4_absorption.csv', delimiter=',', skip_header=2) # x: eV y: um
Be_absorption = np.genfromtxt(data_folder + 'Be_absorption.csv', delimiter=',', skip_header=2) # x: eV, y: um
Al_absorption = np.genfromtxt(data_folder + 'Al_absorption.csv', delimiter=',', skip_header=2) # x: eV, y: um

def thermal_spec(ene, kT, norm):
    '''Return a normalized thermal spectrum
    
    Parameters
    ----------
    ene : float or array
        Energy in eV
    kT : float
        Temperature in keV
    norm : float
        Peak of the spectrum , in photons/cm^2/s/eV'''
    return norm * ene ** 3 / (np.exp(ene / kT) - 1) / (1.42143 * kT ** 3)

def power_law(tot_flux, index, ene_low=100., ene_high=25000.):
    '''Return a normalized power law spectrum
    
    Parameters
    ----------
    tot_flux : float
        Total flux in the energy range, in erg/s/cm^2
    index : float
        Power law index
    ene_low : float
        Lower bound of the energy range, in eV
    ene_high : float
        Upper bound of the energy range, in eV
        
    Returns
    -------
    pow_spec : array
        Spectrum (x: energy in eV, y: spectral flux in phot/cm^2/s/eV)'''

    energies = np.logspace(np.log10(ene_low), np.log10(ene_high), num=300)
    spectral_fluxes = energies ** (-index)
    norm_factor = tot_flux / np.trapz(spectral_fluxes, energies)
    spectral_fluxes *= norm_factor
    band_500_to_10000 = np.logspace(np.log10(500), np.log10(10000), num=300)
    energy_500_to_10000 = np.trapz(spectral_fluxes, band_500_to_10000)
    band_300_to_2000 = np.logspace(np.log10(300), np.log10(2000), num=300)
    energy_300_to_2000 = np.trapz(spectral_fluxes, band_300_to_2000)
    # print("Flux between 0.5 and 10 keV: ", format(energy_500_to_10000, '3.2e'), " erg/s/cm^2")
    # print("Flux between 0.3 and 2 keV: ", format(energy_300_to_2000, '3.2e'), " erg/s/cm^2")
    # Divide by the photon energy, in erg
    spectral_fluxes = spectral_fluxes / (energies * 1.60218e-12)
    pow_spec = np.array([energies, spectral_fluxes]).T
    return pow_spec

def calculate_qe(layers, depletion_depth, ene_low=100., ene_high=25000.):
    """
    Calculate the quantum efficiency of a CCD/CMOS sensor for a given energy range.

    Parameters
    ----------
    layers: list of dictionaries
        Each dictionary has keys 'thickness' (um) and 'material' (string)
    depletion_depth: float
        The depletion depth of the sensor in um
    ene_low: 
    """
    energies = np.logspace(np.log10(ene_low), np.log10(ene_high), num=300)
    # Calculate the probability that the photon is not absorbed in the outer layers
    transmission = np.ones_like(energies)
    for layer in layers:
        if layer['material'] == 'Si':
            abs_lengths = np.interp(energies, Si_absorption[:, 0], Si_absorption[:, 1])
            transmission *= np.exp(-layer['thickness'] / abs_lengths)
        elif layer['material'] == 'SiO2':
            abs_lengths = np.interp(energies, SiO2_absorption[:, 0], SiO2_absorption[:, 1])
            transmission *= np.exp(-layer['thickness'] / abs_lengths)
        elif layer['material'] == 'Si3N4':
            abs_lengths = np.interp(energies, Si3N4_absorption[:, 0], Si3N4_absorption[:, 1])
            transmission *= np.exp(-layer['thickness'] / abs_lengths)
        elif layer['material'] == 'Be':
            abs_lengths = np.interp(energies, Be_absorption[:, 0], Be_absorption[:, 1])
            transmission *= np.exp(-layer['thickness'] / abs_lengths)
        elif layer['material'] == 'Al':
            abs_lengths = np.interp(energies, Al_absorption[:, 0], Al_absorption[:, 1])
            transmission *= np.exp(-layer['thickness'] / abs_lengths)
        else:
            raise ValueError('Material not recognized.')
    # Calculate the absorption length for the depletion depth
    si_abs_lengths = np.interp(energies, Si_absorption[:, 0], Si_absorption[:, 1])
    qe = transmission * (1 - np.exp(-depletion_depth / si_abs_lengths))
    # Pair up energies with quantum efficiencies
    qe_array = np.vstack((energies, qe)).T
    return qe_array

def snr(eff_area, open_frac, source_spec, bkg_spec, exposure_time, fov):
    '''Calculate the SNR for given x-ray source and background spectrum
    Parameters
    ----------
    eff_area: array
        Detector effective area spectrum (x: energy in eV, y: cm^2)
    open_frac: float
        Open fraction of the coded mask
    source_spec: array
        Source spectrum (x: energy in eV, y: spectral flux density, in counts/s/cm^2/eV)
    bkg_spec: array
        Background spectrum (x: energy in eV, y: spectral flux density, in counts/s/cm^2/eV/sr)
    exposure_time: float
        Exposure time in seconds
    fov: float
        Field of view in steradians
    '''
    # Calculate the source flux. First line up the source and effective area energies
    source_spec_interp = np.interp(eff_area[:, 0], source_spec[:, 0], source_spec[:, 1])
    source_flux = np.trapz(source_spec_interp * eff_area[:, 1], eff_area[:, 0]) # counts/s
    bkg_spec_interp = np.interp(eff_area[:, 0], bkg_spec[:, 0], bkg_spec[:, 1])
    bkg_flux = fov * np.trapz(bkg_spec_interp * eff_area[:, 1], eff_area[:, 0]) # counts/s
    # print(source_flux, bkg_flux)
    snr = source_flux * np.sqrt(exposure_time * open_frac) / np.sqrt(source_flux + open_frac * bkg_flux / (1 - open_frac))
    return snr

def collimator_snr(eff_area, source_spec, bkg_spec, exposure_time, bkg_fov):
    # Calculate the source flux. First line up the source and effective area energies
    source_spec_interp = np.interp(eff_area[:, 0], source_spec[:, 0], source_spec[:, 1])
    source_flux = np.trapz(source_spec_interp * eff_area[:, 1], eff_area[:, 0]) # counts/s
    print('Source Flux: ', format(source_flux, '3.2e'), ' counts/s')
    bkg_spec_interp = np.interp(eff_area[:, 0], bkg_spec[:, 0], bkg_spec[:, 1])
    bkg_flux = bkg_fov * np.trapz(bkg_spec_interp * eff_area[:, 1], eff_area[:, 0]) # counts/s
    signal = source_flux * exposure_time
    noise = np.sqrt(signal + bkg_flux * exposure_time)
    snr = signal / noise
    return snr

def rel_eff_area_map(mask_size, det_size, height, use_max_angle=True, max_angle=50*np.pi/180, resolution=50, disp_map=False):
    '''Calculate the relative effective area across the PCFV
    Parameters
    ----------
    mask_size: float
        Size of the coded mask in cm. Assumes square mask.
    det_size: float
        Size of the detector in cm. Assumes square detector.
    height: float
        Distance between the mask and detector in cm
    resolution: int
        Number of pixels per side to split the PCFV into
    disp_map: bool
        Whether to display the effective area map
    
    Returns
    -------
    rel_eff_area_array: array
        2D array of the relative effective area across the PCFV
    eff_fov: float
        Effective field of view in square degrees
    '''
    # Calculate the PCFV half-angle
    pcfv_angle = np.arctan((mask_size + det_size) / 2 / height) # radians
    pcfv_deg = pcfv_angle * 180 / np.pi
    # Calculate the FCFV half-angle
    fcfv_angle = np.arctan((mask_size - det_size) / 2 / height) # radians
    rel_eff_area_array = np.zeros((resolution, resolution))
    if use_max_angle:
        angles = np.linspace(-max_angle, max_angle, resolution)
    else:
        angles = np.linspace(-pcfv_angle, pcfv_angle, resolution)
    for i in range(resolution):
        angle_x = angles[i]
        # Calculate the x fraction of the detector that is coded by the mask
        if np.abs(angle_x) <= fcfv_angle:
            coded_frac_x = 1
        elif np.abs(angle_x) >= pcfv_angle:
            coded_frac_x = 0
        else:
            # Find the x position at which the light ray would hit the detector,
            # if it grazes the edge of the mask
            x_intersect = mask_size / 2 - np.abs(height * np.tan(angle_x))
            coded_frac_x = (x_intersect + det_size / 2) / det_size
        for j in range(resolution):
            angle_y = angles[j]
            if np.abs(angle_y) <= fcfv_angle:
                coded_frac_y = 1
            elif np.abs(angle_y) >= pcfv_angle:
                coded_frac_y = 0
            else:
                y_intersect = mask_size / 2 - np.abs(height * np.tan(angle_y))
                coded_frac_y = (y_intersect + det_size / 2) / det_size
            # Calculate the effective area at this point
            rel_eff_area_array[i, j] = coded_frac_x * coded_frac_y


    if disp_map:
        if use_max_angle:
            max_angle_deg = max_angle * 180 / np.pi
            plot_extent = [-max_angle_deg, max_angle_deg, -max_angle_deg, max_angle_deg]
        else:
            plot_extent = [-pcfv_deg, pcfv_deg, -pcfv_deg, pcfv_deg]
        plt.imshow(rel_eff_area_array, extent=plot_extent)
        plt.xlabel('Angle 1 (deg)')
        plt.ylabel('Angle 2 (deg)')
        # Label the colorbar
        plt.colorbar(label='Relative effective area')
        plt.show()

    # Calculate the effective field of view. This is useful for calculating the grasp of the telescope.
    angles_deg = angles * 180 / np.pi
    angle_box_area = (angles_deg[1] - angles_deg[0]) ** 2
    # Integrating the relative A_eff over solid angle gives an "effective FOV"
    eff_fov = np.sum(rel_eff_area_array) * angle_box_area
    return angles_deg * 180 / np.pi, rel_eff_area_array, eff_fov    

if __name__ == '__main__':
    energy_low = 100. # eV
    energy_high = 20000. # eV
    energy_step = 10. # eV
    ccid20_layers = [{'thickness': 25, 'material': 'Be'},
                     {'thickness': 0.06, 'material': 'SiO2'},
                     {'thickness': 0.045, 'material': 'Si3N4'},
                     {'thickness': 0.360, 'material': 'Si'},
                     {'thickness': 0.360, 'material': 'SiO2'}]
    ccid20_depth = 30 # um
    ccid20_qe = calculate_qe(ccid20_layers, ccid20_depth, energy_low, energy_high)

    ccid80_layers = [{'thickness': 0.16, 'material': 'Al'},
                     {'thickness': 25 * 10 ** -4, 'material': 'SiO2'}]
    ccid80_depth = 100 # um
    ccid80_qe = calculate_qe(ccid80_layers, ccid80_depth, energy_low, energy_high)

    plot_qe = False
    if plot_qe:
        plt.plot(ccid20_qe[:, 0], ccid20_qe[:, 1], label='FSI CCID-20, Be OBF')
        plt.plot(ccid80_qe[:, 0], ccid80_qe[:, 1], label='BSI CCID-80, Al OBF')
        plt.ylim([0, 1])
        plt.xlabel('Energy (eV)')
        plt.ylabel('Quantum Efficiency')
        plt.legend()
        plt.show()

    ccid_area = 18.87 # cm^2. Same for CCID-20 and CCID-80
    hete_open_frac = 0.2 # Open fraction of the coded mask
    tess_geo_open_frac = 0.2

    hete_eff_area = copy.copy(ccid20_qe)
    hete_eff_area[:, 1] *= (hete_open_frac * 4 * ccid_area) # 4 CCIDs, but only half work because of OBF loss
    tess_geo_eff_area = copy.copy(ccid80_qe)
    tess_geo_eff_area[:, 1] *= (tess_geo_open_frac * 4 * ccid_area) # 4 CCIDs
    swift_eff_area = np.genfromtxt(data_folder + 'SwiftXRT_Aeff.csv', delimiter=',', skip_header=1) # x: eV, y: cm^2
    erosita_eff_area = np.genfromtxt(data_folder + 'eRosita_Aeff.csv', delimiter=',') # x: eV, y: cm^2
    nicer_eff_area = np.genfromtxt(data_folder + 'nicer_aeff.csv', delimiter=',') # x: keV, y: cm^2
    nicer_eff_area[:, 0] *= 1000 # Convert keV to eV
    nicer_eff_area[:, 1] /= 56 # Consider just one module

    eff_fov_10cm = rel_eff_area_map(10, 6.144, 9.5)[2]
    eff_fov_14cm = rel_eff_area_map(14, 6.144, 9.5)[2]
    hete_grasp = copy.copy(hete_eff_area)
    hete_grasp[:, 1] *= eff_fov_10cm
    tess_geo_grasp = copy.copy(tess_geo_eff_area)
    tess_geo_grasp[:, 1] *= eff_fov_10cm
    tess_geo_grasp_max = copy.copy(tess_geo_eff_area)
    tess_geo_grasp_max[:, 1] *= eff_fov_14cm
    
    einstein_probe_grasp = np.genfromtxt(data_folder + 'einstein_probe_grasp.csv', delimiter=',')
    einstein_probe_grasp[:, 0] *= 1000 # Convert keV to eV
    
    plot_grasp = False
    if plot_grasp:
        plt.plot(einstein_probe_grasp[:, 0], einstein_probe_grasp[:, 1], label='Einstein Probe')
        plt.plot(tess_geo_grasp[:, 0], tess_geo_grasp[:, 1], label='TESS-GEO SXC (r=50%, d=10 cm)')
        plt.plot(tess_geo_grasp_max[:, 0], tess_geo_grasp_max[:, 1], label='TESS-GEO SXC (r=50%, d=14 cm)')
        plt.xlabel('Energy (eV)')
        plt.ylabel('Grasp (cm$^2$ deg$^2$)')
        plt.yscale('log')
        plt.legend()
        plt.show()

    # Read the Crab Nebula spectrum. Spectrum is E^2 dN/dEdAdt, in TeV/cm^2/s.
    crab_spec = np.genfromtxt(data_folder + 'crab_spec.csv', delimiter=',', skip_header=1)
    crab_spec[:,0] *= 10 ** 9 # Convert to eV
    # Only consider the x-ray part of the spectrum, where energy is between 100 and 30000 eV
    # crab_spec = crab_spec[(crab_spec[:, 0] > 100) & (crab_spec[:, 0] < 30000)]
    crab_spec = crab_spec[(crab_spec[:, 0] > 1000) & (crab_spec[:, 0] < 6000)]
    # plt.plot(crab_spec[:, 0], crab_spec[:, 1])
    # plt.yscale('log')
    # plt.xscale('log')
    # plt.xlabel('Energy (eV)')
    # plt.ylabel('Photons/cm$^2$/s/eV')
    # plt.show()
    # Convert the spectrum to photons/cm^2/s/eV
    crab_spec[:, 1] = crab_spec[:, 1] * 10 ** 12 / crab_spec[:, 0] ** 2
    crab_spec[:, 1] = (crab_spec[:, 0] / 1000) ** -2 * 0.01

    # Load x-ray background spectrum. x: E (keV); y: nu*Fnu (keV^2/cm^2/s/sr/keV)
    bkg_spec = np.genfromtxt(data_folder + 'xray_bkg.csv', delimiter=',', skip_header=1)
    bkg_spec[:, 1] /= (bkg_spec[:, 0] ** 2) # Convert to phot/cm^2/s/keV/sr
    bkg_spec[:, 1] /= 1000 # Convert to phot/cm^2/s/eV/sr
    bkg_spec[:, 0] *= 1000 # Convert to eV

    source_spec = power_law(5e-12, 2, ene_low=200, ene_high=20000)

    def get_limiting_flux(collimator_area, fov, exposure_time=900, snr=5, flux_guess=1e-12):
        def snr_diff(flux):
            source_spec = power_law(flux, 2, ene_low=200, ene_high=20000)
            snr_result = collimator_snr(collimator_area, source_spec, bkg_spec, exposure_time, fov)
            return snr_result - snr
        sol = scipy.optimize.root(snr_diff, flux_guess)
        return sol.x

    # collimator_eff_area = copy.copy(ccid80_qe)
    wall_thickness = 0.2 # mm
    collimator_height = 86 # mm
    num_points = 1
    limiting_flux_array = np.zeros(num_points)
    slat_width_array = np.linspace(0.2, 3, num_points)
    slat_width_array = np.array([2])
    fov_array = np.zeros(num_points)
    throughput_array = np.zeros(num_points)
    snr_e11 = np.zeros(num_points)
    for i, slat_width in enumerate(slat_width_array):
        fov = np.arctan(slat_width / collimator_height) ** 2
        fov_array[i] = np.sqrt(fov * 3282.8)
        collimator_throughput = ((slat_width) / (slat_width + wall_thickness)) ** 2
        throughput_array[i] = collimator_throughput
        collimator_eff_area = copy.copy(ccid80_qe)
        collimator_eff_area[:, 1] *= (4 * ccid_area * collimator_throughput) # 4 CCIDs
        limiting_flux_array[i] = get_limiting_flux(collimator_eff_area, fov, flux_guess=1e-12)
        source_flux_e10 = power_law(1e-10, 2, ene_low=200, ene_high=20000)
        snr_e11[i] = collimator_snr(collimator_eff_area, source_flux_e10, bkg_spec, 900, fov)
    print(snr_e11)
    fig, (ax1, ax3) = plt.subplots(2, 1, figsize=(8, 6))
    ax2 = ax1.twinx()
    ax1.plot(slat_width_array, limiting_flux_array, 'g-')
    ax2.plot(slat_width_array, fov_array, 'b-')
    ax1.set_ylabel('Limiting Flux (erg/s/cm$^2$)', color='g')
    ax2.set_ylabel('1D Field of View (deg)', color='b')
    ax4 = ax3.twinx()
    ax3.plot(slat_width_array, snr_e11, 'r-')
    ax4.plot(slat_width_array, throughput_array, 'y-')
    ax3.set_xlabel('Collimator Slit Width (mm)')
    ax3.set_ylabel('SNR (1e-10 erg/s/cm$^2$)', color='r')
    ax4.set_ylabel('Collimator Throughput', color='y')
    fig.subplots_adjust(right=0.85)
    plt.show()
    # plt.plot(slat_width_array, limiting_flux_array)
    # plt.xlabel('Slat Width (mm)')
    # plt.ylabel('Limiting Flux in 900 s (erg/s/cm^2)')
    # plt.show()
    # plt.plot(collimator_eff_area[:, 0], collimator_eff_area[:, 1])
    # plt.xlabel('Energy (eV)')
    # plt.ylabel('Effective Area (cm^2)')
    # plt.show()
    # Set collimator eff area to 50 to compare w/ George's results
    # collimator_eff_area[:, 1] = 50 * 0.6

    # print(snr(tess_geo_eff_area, 0.2, source_spec, bkg_spec, 900, 0.9))

    nicer_fov = 2.54 * 10 ** -6
    # four_sq_deg = 0.001218
    # print(collimator_snr(collimator_eff_area, source_spec, bkg_spec, 900, four_sq_deg / 4 * 3))
    short_concentrator_eff_area = copy.copy(nicer_eff_area)
    short_concentrator_eff_area[:, 1] *= 1.7
    print(get_limiting_flux(short_concentrator_eff_area, nicer_fov, flux_guess=1e-12))
    source_flux_e10 = power_law(1e-10, 2, ene_low=200, ene_high=20000)
    print(collimator_snr(short_concentrator_eff_area, source_flux_e10, bkg_spec, 900, nicer_fov))
    
    # # # GRB spectrum. x: E (eV); y: E^2*N(E) (erg/cm^2/s)
    # # grb_spec = np.genfromtxt(data_folder + 'grb_spectrum.csv', delimiter=',', skip_header=1)
    # # # Convert to phot/cm^2/s/eV
    # # grb_spec[:, 1] /= (grb_spec[:, 0] ** 2 * 1.60218e-12)
    # # energies = np.arange(150, 20000, 50)
    # # grb_spec_interp = np.interp(energies, grb_spec[:, 0], grb_spec[:, 1])
    # # grb_num_phot = np.trapz(grb_spec_interp, energies)
    # # print(grb_num_phot)
    # # grb_flux = np.trapz(grb_spec_interp * energies * 1.60218e-12, energies)
    # # print(grb_flux)

    collimator_throughput = (2 / 2.2) ** 2
    collimator_eff_area = copy.copy(ccid80_qe)
    collimator_eff_area[:, 1] *= collimator_throughput * 4 * ccid_area # 4 CCIDs

    plot_eff_area = False
    if plot_eff_area:
    #     plt.plot(hete_eff_area[:, 0], hete_eff_area[:, 1], label='HETE SXC (r=0.2)')
    #     plt.plot(tess_geo_eff_area[:, 0], tess_geo_eff_area[:, 1], label='TESS-GEO SXC (r=0.2)')
    #     plt.plot(tess_geo_eff_area[:, 0], 2.5 * tess_geo_eff_area[:, 1], label='TESS-GEO SXC (r=0.5)')
        plt.plot(collimator_eff_area[:, 0], collimator_eff_area[:, 1], label='Collimator (2 modules; 2mm slats)')
        plt.plot(nicer_eff_area[:, 0], 1.7 * nicer_eff_area[:, 1], label='Concentrator (8 modules; 50cm focal length)')
        plt.plot(swift_eff_area[:, 0], swift_eff_area[:, 1], label='Swift XRT')
        # plt.plot(erosita_eff_area[:, 0], erosita_eff_area[:, 1], label='eRosita')
        # plt.ylim(0.1, 2000)
        # plt.xlim(100, 2000)
        # plt.yscale('log')
        plt.xlabel('Energy (eV)')
        plt.ylabel('Effective Area (cm^2)')
        plt.legend()
        plt.show()
    