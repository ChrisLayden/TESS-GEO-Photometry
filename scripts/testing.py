import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.axes_grid1 import make_axes_locatable

plt.rcParams["figure.figsize"] = [7.50, 3.50]
plt.rcParams["figure.autolayout"] = True

fig = plt.figure()

ax = fig.add_subplot(111)
div = make_axes_locatable(ax)
cax = div.append_axes('right', '5%', '5%')
data = np.random.rand(5, 5)
im = ax.imshow(data)
cb = fig.colorbar(im, cax=cax)

def animate(i):
   cax.cla()
   data = np.random.rand(5, 5)
   im = ax.imshow(data)
   fig.colorbar(im, cax=cax)
   
ani = animation.FuncAnimation(fig, animate, frames=10)

plt.show()