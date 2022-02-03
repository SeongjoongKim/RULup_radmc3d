from matplotlib.colors import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.gridspec as gridspec
from astropy.io import fits
from astropy import units as u
from matplotlib.patches import Ellipse, Rectangle, Circle
from astropy.wcs import WCS
from matplotlib.offsetbox import (AnchoredOffsetbox, AuxTransformBox, DrawingArea, TextArea, VPacker)#from Model_setup_subroutines import *
from gofish import imagecube
import argparse
import os

# =======================================================================================
# Functions
# =======================================================================================
def collapse_zeroth(vel, data, rms,i_start, i_end):
    dvel = np.diff(vel).mean()
    npix = np.sum(data != 0.0, axis=0)
    mom0 = np.trapz(data[i_start:i_end],dx=dvel,axis=0)
    npix = (i_end-i_start)#*np.ones(mom0.shape)
    dmom0 = dvel * rms * npix**0.5 * np.ones(mom0.shape)
    return mom0, dmom0


def collapse_first(vel, data, rms, N_cut, i_start, i_end):
    dvel = np.diff(vel).mean()
    data=data[i_start:i_end,:,:]
    vpix = vel[i_start:i_end, None, None] * np.ones(data.shape)
    weights = 1e-10 * np.random.rand(data.size).reshape(data.shape)
    weights = np.where(data >= N_cut*rms, abs(data), weights)
    mom1 = np.average(vpix, weights=weights, axis=0)
    dmom1 = (vpix - mom1[None, :, :]) * rms / np.sum(weights, axis=0)
    dmom1 = np.sqrt(np.sum(dmom1**2, axis=0))
    npix = np.sum(data >= N_cut*rms, axis=0)
    mom1 = np.where(npix >= 5.0, mom1, 0)
    dmom1 = np.where(npix >= 5.0, dmom1, 0)
    return mom1, dmom1


def collapse_second(vel, data, rms, N_cut, i_start, i_end):
    dvel = np.diff(vel).mean()
    data_mom1=data
    data=data[i_start:i_end,:,:]
    vpix = vel[i_start:i_end, None, None] * np.ones(data.shape)
    weights = 1e-10 * np.random.rand(data.size).reshape(data.shape)
    weights = np.where(data >= N_cut*rms, abs(data), weights)
    mom1 = collapse_first(vel, data_mom1, rms, N_cut, i_start, i_end)[0]
    mom1 = mom1[None, :, :] * np.ones(data.shape)
    mom2 = np.sum(weights * (vpix - mom1)**2, axis=0) / np.sum(weights, axis=0)
    mom2 = np.sqrt(mom2)
    dmom2 = ((vpix - mom1)**2 - mom2**2) * rms / np.sum(weights, axis=0)
    dmom2 = np.sqrt(np.sum(dmom2**2, axis=0)) / 2. / mom2
    npix = np.sum(data >= N_cut*rms, axis=0)
    mom2 = np.where(npix >= 5.0, mom2, 0)
    dmom2 = np.where(npix >= 5.0, dmom2, 0)
    return mom2, dmom2
    
    
# =======================================================================================
# Parameter setup
# =======================================================================================
parser = argparse.ArgumentParser()
parser.add_argument('-mole', default='None', type=str, help='The molecular line name. Default is None.')
parser.add_argument('-rms', default=1e-3, type=float, help='The 1 sigma rms level. Default is 1e-3.')
parser.add_argument('-cut', default=3.0, type=float, help='Sigma cut for M1 and M2 calculation. Default is 3.')
parser.add_argument('-tname', default='fiducial', type=str, help='Test name. Default is fiducial.')
args = parser.parse_args()

mole = args.mole
RMS = args.rms
sig_cut = args.cut
tname = args.tname

# =======================================================================================
# Disk parameters & radii setup
# =======================================================================================
inc = 25.0
DPA = 121.0; PA_min = -180.0; PA_max = 180.0
dxc = 0.00; dyc = 0.00
z0 = 0.00;psi = 1.0; z1 = 0.0; phi = 1.0

# =======================================================================================
# No wind model
# =======================================================================================
fdir = '/Users/kimsj/Documents/RU_Lup/Fin_fits/'
fitsname = mole+'_selfcal_wc_matched_cube500.fits'  #
hdu = fits.open(fdir+fitsname)[0]   # Read the fits file: header + data
data = hdu.data#[0,0,:,:]      # Save the data part
nx = hdu.header['NAXIS1']; ny = hdu.header['NAXIS2']; nv = hdu.header['NAXIS3']  # Set up axis lengths
#if len(data.shape) == 3: data = data.reshape(-1,nv,ny,nx)    # Match the data shape to (1,nv, ny, nx)
#data = np.swapaxes(data, 0, 2); data = np.swapaxes(data, 1, 2) # reshape the data in (nx,ny,nv,1)
#print(data.max(),data.min())
if hdu.header['CTYPE3'] == 'FREQ':
    f0=hdu.header['CRVAL3']*1e-9
    df=hdu.header['CDELT3']*1e-9
    restfreq=hdu.header['RESTFRQ']*1e-9
    c=299792.458 # speed of light (km/s)
    v0=-(f0-restfreq)*c/restfreq
    dv=-df*c/restfreq
if hdu.header['CTYPE3'] == 'VELO-LSR':
    v0 = hdu.header['CRVAL3']
    dv = hdu.header['CDELT3']

#dv = hdu.header['CDELT3'] ; v_ref = hdu.header['CRVAL3']  # delta_V in km/s
velax = np.arange(nv)*dv + v0
# Set continuum level
data_cl = np.zeros((ny,nx))
data_cr = np.zeros((ny,nx))
for i in range(20):
    data_cl += data[0,i,:,:]
    data_cr += data[0,nv-1-i,:,:]
data_cl /= 20.; data_cr /= 20.
data_c = (data_cl+data_cr)*0.5
peak_snr = np.zeros((ny,nx))
for i in range(nx):
    for j in range(ny):
        peak_snr[j,i] = (data[0,:,j,i]-data_c[j,i]).max()/RMS
        
med_snr = peak_snr.min()
assert med_snr < sig_cut, 'Sigma cut ({:5.2f}) is smaller than minimum value of Max. SNR of the cube ({:5.2f})'.format(sig_cut,med_snr)
    
# Momemnt map calculations
M0, dM0 = collapse_zeroth(velax, data[0,:,:,:], 0, 0, nv)
M1, dM1 = collapse_first(velax, data[0,:,:,:]-data_c, RMS, sig_cut, 0, nv)
M2, dM2 = collapse_second(velax, data[0,:,:,:]-data_c, RMS, sig_cut, 0, nv)

# =======================================================================================
# wind model
# =======================================================================================
fdir = '/Users/kimsj/Documents/RADMC-3D/radmc3d-2.0/RU_Lup_test/Automatics/Fin_script/fiducial_wind/'
if len(tname.split('_')) == 1:
    tname_wind = tname+'_wind'
else:
    tname_wind = tname #tname.split('_')[0]+'_wind_'+tname.split('_')[1]
fitsname = 'RULup_'+mole+'_'+tname_wind+'_bmaj51.fits'  #
hduw = fits.open(fdir+fitsname)[0]   # Read the fits file: header + data
dataw = hduw.data#[0,0,:,:]      # Save the data part
nxw = hduw.header['NAXIS1']; nyw = hduw.header['NAXIS2']; nvw = hduw.header['NAXIS3']  # Set up axis lengths
#if len(data.shape) == 3: data = data.reshape(-1,nv,ny,nx)    # Match the data shape to (1,nv, ny, nx)
#data = np.swapaxes(data, 0, 2); data = np.swapaxes(data, 1, 2) # reshape the data in (nx,ny,nv,1)
#print(data.max(),data.min())
dvw = hduw.header['CDELT3'] ; v_refw = hduw.header['CRVAL3']  # delta_V in km/s
velaxw = np.arange(nvw)*dvw + v_refw
# Set continuum level
dataw_c = np.zeros((ny,nx))
for i in range(10):
    dataw_c += dataw[i,:,:]
dataw_c /= 10.
# Moment map calculations
M0w, dM0w = collapse_zeroth(velaxw, dataw, 0, 0, nvw)
M1w, dM1w = collapse_first(velaxw, dataw-dataw_c, 1e-6, 3, 0, nvw)
M2w, dM2w = collapse_second(velaxw, dataw-dataw_c, 1e-6, 3, 0, nvw)

del_M1 = M1 - M1w - 4.5
del_M2 = M2 - M2w

# Set additional contours & Beam size
wcs = WCS(hdu.header).slice([nx,ny])                         # calculate RA, Dec
bmaj=hdu.header['BMAJ']; bmin=hdu.header['BMIN']; bpa=hdu.header['BPA'] # Read beam parameter
bmaj=(bmaj*u.degree).to(u.arcsec).value; bmin=(bmin*u.degree).to(u.arcsec).value # Convert from degree to arcsec
cell=abs(hdu.header['CDELT2']); Cell=(cell*u.degree).to(u.arcsec).value # Read pixel size in arcsec
beam = Ellipse((50., 50.), bmaj*100, bmin*100, angle=90.+bpa, edgecolor='white', facecolor='black')
disk = Ellipse((250,250), 240./150.*100, 240./150.*100*np.cos(inc*np.pi/180.), angle=31., edgecolor='white', facecolor='none',ls='--')  # Keplerian disk
ring = Ellipse((250,250), 520./150.*100, 520./150.*100*np.cos(inc*np.pi/180.), angle=31., edgecolor='black', facecolor='none',ls='--')   # Outer envelope boundary
# Moment 1 map -----------------------------------------------
fig=plt.figure(num=None, figsize=(8,6), dpi=100, facecolor='w', edgecolor='k')
ax = fig.add_subplot(projection=wcs)
im=ax.imshow(del_M1,origin='lower',vmax=1.0,vmin=-0.5,cmap="jet")
ax.set_xlabel('RA',fontsize=15)
ax.set_ylabel('Dec',fontsize=15)
#ax.text(0.95, 0.95, freq[i], ha='right', va='top', transform=ax.transAxes, color="w",fontsize=15)
ax.tick_params(axis='both',which='both',length=4,width=1,labelsize=12,direction='in',color='w')
ax.add_patch(beam)
ax.add_patch(disk)
ax.add_patch(ring)
#ax.margins(x=-0.375,y=-0.375)
cbar=plt.colorbar(im, shrink=0.9)
cbar.set_label(r'$\mathrm{km\ s^{-1}}$', size=10)
plt.savefig('./moments_maps/RULup_'+mole+'_Obs-fiducial_wind_all_M1.pdf', bbox_inches='tight', pad_inches=0.1,dpi=100)
#plt.show()
plt.clf()


# Set additional contours & Beam size
beam = Ellipse((50., 50.), bmaj*100, bmin*100, angle=90.+bpa, edgecolor='white', facecolor='black')
disk = Ellipse((250,250), 240./150.*100, 240./150.*100*np.cos(inc*np.pi/180.), angle=31., edgecolor='white', facecolor='none',ls='--')  # Keplerian disk
ring = Ellipse((250,250), 520./150.*100, 520./150.*100*np.cos(inc*np.pi/180.), angle=31., edgecolor='cyan', facecolor='none',ls='--')   # Outer envelope boundary
# Moment 2 map -----------------------------------------------
fig=plt.figure(num=None, figsize=(8,6), dpi=100, facecolor='w', edgecolor='k')
ax = fig.add_subplot(projection=wcs)
im=ax.imshow(del_M2,origin='lower',vmax=0.5,vmin=-0.5,cmap="jet")
ax.set_xlabel('RA',fontsize=15)
ax.set_ylabel('Dec',fontsize=15)
#ax.text(0.95, 0.95, freq[i], ha='right', va='top', transform=ax.transAxes, color="w",fontsize=15)
ax.tick_params(axis='both',which='both',length=4,width=1,labelsize=12,direction='in',color='w')
ax.add_patch(beam)
ax.add_patch(disk)
ax.add_patch(ring)
#ax.margins(x=-0.375,y=-0.375)
cbar=plt.colorbar(im, shrink=0.9)
cbar.set_label(r'$\mathrm{(km\ s^{-1})^{2}}$', size=10)
plt.savefig('./moments_maps/RULup_'+mole+'_Observation_Obs-fiducial_wind_all_M2.pdf', bbox_inches='tight', pad_inches=0.1,dpi=100)
#plt.show()
plt.clf()

