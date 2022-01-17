from matplotlib.colors import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.patches import Ellipse, Rectangle, Circle
from matplotlib.colors import LogNorm
from astropy.utils.data import get_pkg_data_filename
from astropy.wcs import WCS
from matplotlib.offsetbox import (AnchoredOffsetbox, AuxTransformBox, DrawingArea, TextArea, VPacker)
import glob
from astropy.io import fits
from astropy import units as u
#from scipy import stats
#import bettermoments.collapse_cube as bm
#from Model_setup_subroutines import *
import argparse

# Constants
c = 2.99792458e5 # [km/s]
au  = 1.49598e13     # Astronomical Unit       [cm]
pc  = 3.08572e18     # Parsec                  [cm]

# Parser the arguments
parser = argparse.ArgumentParser()
parser.add_argument('-file', default='None', type=str, help='Input test title')
args = parser.parse_args()
test = args.file

# Reading the observational profiles -------------------------------------------------------
#DSHARP data
rad_obs, cont336_obs = np.loadtxt('./Obs_profiles/RULup_cont336GHz_matched_final_radial.dat' ,usecols=(0,1),unpack=True)  # length =50
rad2_obs, C12O_obs = np.loadtxt('./Obs_profiles/RULup_12CO_matched_mom0_radial.dat' ,usecols=(0,1),unpack=True)   # length =100  0.095X0.083
# Huang et al. 2020 data
rad_obs, cont220_obs = np.loadtxt('./Obs_profiles/RULup_cont220GHz_matched_final_radial.dat' ,usecols=(0,1),unpack=True)
rad_obs, C13O_21_obs = np.loadtxt('./Obs_profiles/RULup_13CO_2-1_wc_matched_final_radial.dat' ,usecols=(0,1),unpack=True)
rad_obs, C13O_32_obs = np.loadtxt('./Obs_profiles/RULup_13CO_3-2_wc_matched_radial.dat' ,usecols=(0,1),unpack=True)
rad_obs, C18O_21_obs = np.loadtxt('./Obs_profiles/RULup_C18O_2-1_wc_matched_final_radial.dat' ,usecols=(0,1),unpack=True)
rad_obs, C18O_32_obs = np.loadtxt('./Obs_profiles/RULup_C18O_3-2_wc_matched_radial.dat' ,usecols=(0,1),unpack=True)
rad_obs, CN_obs = np.loadtxt('./Obs_profiles/RULup_CN_3-2_wc_matched_radial.dat' ,usecols=(0,1),unpack=True)

# Reading model profiles -------------------------------------------------------
rad_mod, cont336_mod = np.loadtxt('./RULup_cont336GHz_'+test+'_radial.dat' ,usecols=(0,1),unpack=True)  # length =50
rad2_mod, C12O_mod = np.loadtxt('./RULup_12CO_2-1_'+test+'_radial.dat' ,usecols=(0,1),unpack=True)   # length =100  0.095X0.083
# Huang et al. 2020 data
rad_mod, cont220_mod = np.loadtxt('./RULup_cont220GHz_'+test+'_radial.dat' ,usecols=(0,1),unpack=True)
rad_mod, C13O_21_mod = np.loadtxt('./RULup_13CO_2-1_'+test+'_radial.dat' ,usecols=(0,1),unpack=True)
rad_mod, C13O_32_mod = np.loadtxt('./RULup_13CO_3-2_'+test+'_radial.dat' ,usecols=(0,1),unpack=True)
rad_mod, C18O_21_mod = np.loadtxt('./RULup_C18O_2-1_'+test+'_radial.dat' ,usecols=(0,1),unpack=True)
rad_mod, C18O_32_mod = np.loadtxt('./RULup_C18O_3-2_'+test+'_radial.dat' ,usecols=(0,1),unpack=True)
rad_mod, CN_mod = np.loadtxt('./RULup_CN_3-2_'+test+'_radial.dat' ,usecols=(0,1),unpack=True)

# Plot continuums
fig=plt.figure(num=None, figsize=(8,6), dpi=100, facecolor='w', edgecolor='k')
plt.subplot(111)
#plt.title(title)
plt.xlabel('radius [au]')
plt.ylabel('I_cont [Jy/beam]')
plt.xlim([10,300])
plt.ylim([5e-7,1e1])
plt.yscale('log')
plt.xscale('log')
plt.plot(rad_obs, cont336_obs, 'k',label='Obs 336 GHz')
plt.hlines(3*6e-5,10,300,colors='k',linestyles=':')
plt.plot(rad_obs, cont220_obs, 'b',label='Obs 220 GHz')
plt.hlines(3*1.0e-4,10,300,colors='b',linestyles=':')
plt.plot(rad_mod, cont336_mod, 'k',ls='--')
plt.plot(rad_mod, cont220_mod, 'b',ls='--')
plt.legend(prop={'size':12},loc=0)
plt.tick_params(which='both',length=6,width=1.5,direction='in')
plt.tight_layout()
plt.savefig('RULup_'+test+'_continuum_major.pdf')

# Plot lines
fig=plt.figure(num=None, figsize=(8,6), dpi=100, facecolor='w', edgecolor='k')
plt.subplot(111)
#plt.title(title)
plt.xlabel('radius [au]')
plt.ylabel('I_line [Jy/beam km/s]')
plt.xlim([10,300])
plt.ylim([5e-7,1e1])
plt.yscale('log')
plt.xscale('log')
plt.plot(rad2_obs, C12O_obs, 'k',label='Obs 12CO 2-1')
plt.plot(rad2_mod, C12O_mod, 'k',ls='--')
plt.plot(rad_obs, CN_obs, 'r',label='Obs CN 3-2')
plt.plot(rad_mod, CN_mod, 'r',ls='--')
plt.plot(rad_obs, C13O_21_obs, 'g',label='Obs 13CO 2-1')
plt.plot(rad_mod, C13O_21_mod, 'g',ls='--')
plt.plot(rad_obs, C13O_32_obs, 'm',label='Obs 13CO 3-2')
plt.plot(rad_mod, C13O_32_mod, 'm',ls='--')
plt.plot(rad_obs, C18O_21_obs, 'b',label='Obs C18O 2-1')
plt.plot(rad_mod, C18O_21_mod, 'b',ls='--')
plt.plot(rad_obs, C18O_32_obs, 'orange',label='Obs C18O 3-2')
plt.plot(rad_mod, C18O_32_mod, 'orange',ls='--')
plt.tick_params(which='both',length=6,width=1.5,direction='in')
plt.legend(prop={'size':12},loc=0)
plt.tight_layout()
plt.savefig('RULup_'+test+'_line_major.pdf')
