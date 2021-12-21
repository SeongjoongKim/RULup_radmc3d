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
parser.add_argument('-mole', default='None', type=str, help='Target molecular species to plot. Default is None')
parser.add_argument('-bmaj', default='bmaj5', type=str, help='Target beam size. Default is bmaj5, indicating 0.05 arcsec')
args = parser.parse_args()
mole = args.mole
bmaj = args.bmaj

# Reading the observational profiles -------------------------------------------------------
#DSHARP data
obs_dir = './Obs_profiles/'
if mole == 'cont336GHz': obs_file = 'RULup_cont336GHz_matched_final_radial.dat'
if mole == 'cont220GHz': obs_file = 'RULup_cont220GHz_matched_final_radial.dat'
if mole == '12CO_2-1': obs_file = 'RULup_12CO_matched_mom0_radial.dat'
if mole == '13CO-2-1': obs_file = 'RULup_13CO_2-1_wc_matched_final_radial.dat'
if mole == '13CO_3-2': obs_file = 'RULup_13CO_3-2_wc_matched_radial.dat'
if mole == 'C18O_2-1': obs_file = 'RULup_C18O_2-1_wc_matched_final_radial.dat'
if mole == 'C18O_3-2': obs_file = 'RULup_C18O_3-2_wc_matched_radial.dat'
if mole == 'CN_3-2': obs_file = 'RULup_CN_3-2_wc_matched_radial.dat'
r_obs, I_obs = np.loadtxt(obs_dir+obs_file ,usecols=(0,1),unpack=True)  # length =50

# Reading model profiles -------------------------------------------------------
r_f, I_f = np.loadtxt('./fiducial/RULup_'+mole+'_fiducial_'+bmaj+'_radial.dat' ,usecols=(0,1),unpack=True)  # length =50
r_w, I_w = np.loadtxt('./fiducial_wind/RULup_'+mole+'_fiducial_wind_'+bmaj+'_radial.dat' ,usecols=(0,1),unpack=True)  # length =50
r_CNw, I_CNw = np.loadtxt('./fiducial_wind/RULup_'+mole+'_fiducial_wind_CN0.35_Cw1e-4_'+bmaj+'_radial.dat' ,usecols=(0,1),unpack=True)  # length =50
r_CN, I_CN = np.loadtxt('./fiducial_wind/RULup_'+mole+'_fiducial_wind_CN0.35_'+bmaj+'_radial.dat' ,usecols=(0,1),unpack=True)  # length =50
r_th, I_th = np.loadtxt('./fiducial_wind/RULup_'+mole+'_fiducial_wind_Thermal_'+bmaj+'_radial.dat' ,usecols=(0,1),unpack=True)  # length =50

# Plot lines
fig=plt.figure(num=None, figsize=(8,6), dpi=100, facecolor='w', edgecolor='k')
plt.subplot(111)
plt.title(mole)
plt.xlabel('radius [au]')
plt.ylabel('I_line [Jy/beam]')
plt.xlim([10,300])
plt.ylim([5e-7,1e1])
plt.yscale('log')
plt.xscale('log')
plt.plot(r_obs, I_obs, 'k--',label='Obs')
plt.plot(r_f, I_f, 'r',label='Fiducial')
plt.plot(r_w, I_w, 'g',label='Fiducial_wind')
plt.plot(r_th, I_th, 'b',label='No wind +Thermal only')
plt.plot(r_CN, I_CN, 'm--',label=r'Z$_{CN}$ = 0.35',lw=0.8)
plt.plot(r_CNw, I_CNw, 'orange',label=r'Z$_{CN}$ = 0.35 + C$_{w}$ = 1e-4', ls='--',lw=0.8)
plt.tick_params(which='both',length=6,width=1.5,direction='in')
plt.legend(prop={'size':12},loc=0)
plt.tight_layout()
plt.savefig('RULup_'+mole+'_testings_major.pdf')
