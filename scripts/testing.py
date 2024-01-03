# A python file to test out random things

from astropy.io import fits
import numpy as np
import os
import matplotlib.pyplot as plt

dir = '/Users/layden/Library/CloudStorage/Box-Box/clayden7/TESS-GEO Sensor Testing/IMX487/Photon Transfer'
dark_dir = dir + '/Dark Images'
gray_dir = dir + '/Gray Images'
dark_current_dir = dir + '/Dark Current'

for file in os.listdir(gray_dir):
    # If file has '0s', get the time and multiply by 1000
    if '0s' in file:
        time = int(float(file.split('s')[0]) * 1000)
        # Rename file with time + 'ms', then the bit after '0s'
        os.rename(os.path.join(gray_dir, file), os.path.join(gray_dir, str(time) + 'ms' + file.split('0s')[1]))

# time_vals = []
# signal_vals = []
# for file in os.listdir(gray_dir):
#     if not file.endswith('.fits'):
#         continue
#     img = fits.getdata(os.path.join(gray_dir, file)).astype('int')
#     # Remove 'filter7b_' from file name
#     exposure_time = float(file[9:-8])
#     time_vals.append(exposure_time)
#     # Find the median value
#     signal = np.mean(img)
#     signal_vals.append(signal)

# print(time_vals)
# print(signal_vals)
# plt.scatter(time_vals, signal_vals, s=1)
# plt.xlabel('Exposure Time (s)')
# plt.ylabel('Dark Current (ADU)')
# plt.show()
    
# plt.scatter([0,0,1,1,2,2],[0,1,0,1,0,1])
# plt.show()