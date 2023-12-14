'''Tools for adding jitter to an observation.

Functions
---------
jitter_animation : function
    Animate the jittering of an intensity grid.
avg_intensity_to_frame : function
    Convert an intensity grid to an exposure with random noise.
shift_grid : function
    Shift an intensity grid by a given subpixel displacement.
jittered_array : function
    The intensity grid averaged over steps of jittering.
observed_image : function
    The actual image measured by an observatory.
signal_list : function
    A list of signals observed for a given constant spectrum.
multithread_signal_list : function
    A list of average signals and standard deviations observed
    for a given list of spectra.
'''

from multiprocessing import Pool
from functools import partial
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def shift_values(arr, del_x, del_y):
    '''Shift values in an array by a specified discrete displacement.
    
    Parameters
    ----------
    arr : array-like
        The intensity grid.
    del_x : int
        The x displacement, in subpixels.
    del_y : int
        The y displacement, in subpixels.
        
    Returns
    -------
    new_arr : array-like
        The shifted array.
    '''
    N, M = arr.shape
    new_arr = np.zeros_like(arr)
    new_arr[max(del_x, 0):M+min(del_x, 0), max(del_y, 0):N+min(del_y, 0)] = \
        arr[-min(del_x, 0):M-max(del_x, 0), -min(del_y, 0):N-max(del_y, 0)]
    return new_arr

def jittered_array(arr, num_steps, pix_jitter, resolution):
    '''Jitter the values in an array and take the average array.

    Parameters
    ----------
    arr : array-like
        The initial intensity grid.
    num_steps : int
        The number of steps to average over.
    pix_jitter : float
        The RMS jitter, in pixels.

    Returns
    -------
    avg_arr : array-like
        The final intensity grid.
    '''
    # Return error if jitter is too large for the subgrid.
    # Do this if 2.5*sigma is greater than half the subgrid size.
    # This criterion corresponds to 1% of positions being outside the subgrid.
    img_size = float(arr.shape[0])
    if (2.5 * pix_jitter*resolution) > (img_size / 2):
        raise ValueError("Jitter too large for subarray.")
    angles = np.random.uniform(low=0.0, high=2*np.pi, size=num_steps)
    displacements = np.random.normal(scale=pix_jitter * resolution,
                                     size=num_steps)

    # For any displacement that is too large, set it to the maximum allowed.
    displacements[displacements > (img_size / 2)] = img_size / 2

    del_x_list = np.rint(np.cos(angles) * displacements).astype(int)
    del_y_list = np.rint(np.sin(angles) * displacements).astype(int)
    avg_arr = np.zeros_like(arr)
    for i in range(num_steps):
        shifted_arr = shift_values(arr, del_x_list[i], del_y_list[i])
        avg_arr += shifted_arr
    avg_arr /= num_steps
    return avg_arr

# def jitter_animation(initial_grid, exposure_time, jitter):
#     '''Animate the jittering of an intensity grid.

#     Parameters
#     ----------
#     initial_grid : array-like
#         The initial intensity grid.
#     exposure_time : float
#         The total exposure time, in seconds.
#     jitter : float
#         The RMS jitter, in pixels.

#     Returns
#     -------
#     ani : matplotlib.animation.FuncAnimation
#         The animation object.
#     '''
#     num_steps = exposure_time // jitter_time
#     angles = np.random.uniform(low=0.0, high=2*np.pi, size=num_steps)
#     displacements = np.random.normal(scale=jitter * resolution,
#                                      size=num_steps)
#     x_shifts = np.rint(np.cos(angles) * displacements).astype(int)
#     y_shifts = np.rint(np.sin(angles) * displacements).astype(int)
#     initial_frame = initial_grid.reshape((sub_size, resolution, sub_size,
#                                           resolution)).sum(axis=(1, 3))
#     N, M = initial_grid.shape

#     def update(frame):
#         '''Do one step in the animation.'''
#         i = frame
#         shifted_grid = np.zeros_like(initial_grid)
#         shifted_grid[max(x_shifts[i], 0):M+min(x_shifts[i], 0),
#                      max(y_shifts[i], 0):N+min(y_shifts[i], 0)] = \
#             initial_grid[-min(x_shifts[i], 0):M-max(x_shifts[i], 0),
#                          -min(y_shifts[i], 0):N-max(y_shifts[i], 0)]
#         return shifted_grid

#     def stacked_intensity_grid(end_step):
#         grid = np.zeros_like(initial_grid)
#         for step in range(end_step):
#             grid += update(step)
#         return grid / end_step

#     def stacked_image(end_step):
#         grid = np.zeros_like(initial_grid)
#         for step in range(end_step):
#             grid += update(step)
#         image = grid.reshape((sub_size, resolution, sub_size,
#                               resolution)).sum(axis=(1, 3))
#         return image / end_step

#     anim_fig, (anim_ax1, anim_ax2, anim_ax3) = plt.subplots(1, 3)
#     anim_fig.set_size_inches(12, 4)
#     anim_ax1.set_title("Intensity Grid in Step")
#     anim_ax2.set_title("Total Intensity Grid So Far")
#     anim_ax3.set_title("Total Frame So Far")
#     ln1, = [anim_ax1.imshow(initial_grid / num_steps)]
#     ln2, = [anim_ax2.imshow(initial_grid)]
#     ln3, = [anim_ax3.imshow(initial_frame)]
#     anim_images = [ln1, ln2, ln3]

#     def anim_update(frame):
#         shifted_grid = update(frame)
#         stacked_grid = stacked_intensity_grid(frame + 1)
#         stacked_frame = stacked_image(frame + 1)
#         ln1.set_array(shifted_grid / num_steps)
#         ln2.set_array(stacked_grid)
#         ln3.set_array(stacked_frame)
#         return anim_images

#     ani = FuncAnimation(anim_fig, anim_update, frames=range(num_steps),
#                         blit=True)
#     plt.show()
#     return ani

# def multithread_signal_list(observatory, jitter, spectrum_list):
#     '''A list of signals observed for a given list of spectra.

#     Parameters
#     ----------
#     observatory : Observatory
#         The observatory object.
#     obs_duration : float
#         The total observation duration, in seconds.
#     jitter : float
#         The RMS jitter, in pixels.

#     Returns
#     -------
#     means_list : array-like
#         The list of mean signals observed for each spectrum.
#     stds_list : array-like
#         The list of standard deviations of the signals observed
#         for each spectrum.'''
#     processes_pool = Pool(len(spectrum_list))
#     new_func = partial(signal_list, observatory, jitter)
#     out_list = processes_pool.map(new_func, spectrum_list)
#     means_list = np.mean(out_list, axis=1)
#     stds_list = np.std(out_list, axis=1)
#     return means_list, stds_list

if __name__ == '__main__':
    pass
