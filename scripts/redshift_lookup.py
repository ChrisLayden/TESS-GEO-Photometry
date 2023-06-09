'''Tools for calculating redshifts from luminosity distances.

Classes
-------
RedshiftLookup : class
    A class that calculates a table of luminosity distances for
    redshifts out to z = 1.0.  It has a method to interpolate the
    table in order to find a redshift for a given distance.

Functions
---------
lumdist : function
    Numerically intergrates function to calculate luminosity distance.'''

import numpy as np
import matplotlib.pyplot as plt
import constants
from scipy.integrate import quad


def lumdist(z):
    '''Numerically intergrates function to calculate luminosity
    distance.

    Assumes a flat universe.  For now, H0 and Omega_m are hard-coded to
    80 km/s/Mpc and 0.3.  Radiation from early universe is ignored.
    '''
    def E(z, Omega_m):
        # assumes flat universe
        return 1./np.sqrt(
            Omega_m*(1.+z)**3 + (1. - Omega_m)
        )
    H0 = 70*1.e5  # convert to cm/s/Mpc
    Omega_m = 0.3

    # calculate the luminosity distnace, returns in Mpc
    integral = quad(E, 0, z, args=(Omega_m))
    # h0 in km/s/Mpc
    dp = constants.lightspeed/(H0/constants.CM_PER_PARSEC/1.e6)*integral[0]

    return (1+z)*dp/constants.CM_PER_PARSEC/1.e6


class RedshiftLookup(object):
    '''This class will will calculate a table of luminosity distances
    for redshifts out to z = 1.0.  It has a method to interpolate the
    table in order to find a redshift for a given distance.

    '''
    def __init__(self):
        self.redshifts = np.r_[0:1.0:100j]

        # returns these in Mpc
        self.distances = np.array([lumdist(z) for z in self.redshifts])

    def __call__(self, distance):
        return np.interp(distance, self.distances, self.redshifts)


if __name__ == '__main__':
    ztab = RedshiftLookup()
    print('Test:  The redshift for lum dist = 1 Gpc is ~0.204. We get: ',
          ztab(1.e3))

    plt.plot(ztab.redshifts, ztab.distances, 'k.-')
    plt.xlabel('redshift')
    plt.ylabel('Luminosity Distance (Mpc)')
    plt.show()
