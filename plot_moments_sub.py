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
import argparse

# =======================================================================================
# Functions
# =======================================================================================
def collapse_zeroth(vel, data, rms,i_start, i_end):
    dv = np.diff(vel).mean()
    npix = np.sum(data != 0.0, axis=0)
    mom0 = np.trapz(data[i_start:i_end],dx=dv,axis=0)
    npix = (i_end-i_start)#*np.ones(mom0.shape)
    dmom0 = dv * rms * npix**0.5 * np.ones(mom0.shape)
    return mom0, dmom0


def collapse_first(vel, data, rms, N_cut, i_start, i_end):
    dv = np.diff(vel).mean()
    data=data[i_start:i_end,:,:]
    vpix = vel[i_start:i_end, None, None] * np.ones(data.shape)
    weights = 1e-10 * np.random.rand(data.size).reshape(data.shape)
    weights = np.where(data >= N_cut*rms, abs(data), weights)
    M1 = np.average(vpix, weights=weights, axis=0)
    dM1 = (vpix - M1[None, :, :]) * rms / np.sum(weights, axis=0)
    dM1 = np.sqrt(np.sum(dM1**2, axis=0))
    npix = np.sum(data >= N_cut*rms, axis=0)
    M1 = np.where(npix >= 5.0, M1, 0)
    dM1 = np.where(npix >= 5.0, dM1, 0)
    return M1, dM1


def collapse_second(vel, data, rms, N_cut, i_start, i_end):
    dv = np.diff(vel).mean()
    data_M1=data
    data=data[i_start:i_end,:,:]
    vpix = vel[i_start:i_end, None, None] * np.ones(data.shape)
    weights = 1e-10 * np.random.rand(data.size).reshape(data.shape)
    weights = np.where(data >= N_cut*rms, abs(data), weights)
    M1 = collapse_first(vel, data_M1, rms, N_cut, i_start, i_end)[0]
    M1 = M1[None, :, :] * np.ones(data.shape)
    M2 = np.sum(weights * (vpix - M1)**2, axis=0) / np.sum(weights, axis=0)
    M2 = np.sqrt(M2)
    dM2 = ((vpix - M1)**2 - M2**2) * rms / np.sum(weights, axis=0)
    dM2 = np.sqrt(np.sum(dM2**2, axis=0)) / 2. / M2
    npix = np.sum(data >= N_cut*rms, axis=0)
    M2 = np.where(npix >= 5.0, M2, 0)
    dM2 = np.where(npix >= 5.0, dM2, 0)
    return M2, dM2
    
    
# =======================================================================================
# Parameter setup
# =======================================================================================
parser = argparse.ArgumentParser()
parser.add_argument('-mole1', default='None', type=str, help='The molecular line name. Default is None.')
parser.add_argument('-mole2', default='None', type=str, help='The molecular line name. Default is None.')
parser.add_argument('-bmaj', default='bmaj5', type=str, help='The beam size. Default is bmaj5.')
parser.add_argument('-tname', default='fiducial_wind', type=str, help='Test name. Default is fiducial.')
args = parser.parse_args()

mole1 = args.mole1  #'CN_3-2'
mole2 = args.mole2  #'CN_3-2'
b_maj = args.bmaj  #'bmaj51'
tname = args.tname  #'fiducial'

inc = 25.0
r_min = 0.35;  r_max = 0.45
DPA = 121.0; PA_min = 80; PA_max = 100
dxc = 0.00; dyc = 0.00
z0 = 0.00;psi = 1.0; z1 = 0.0; phi = 1.0

# =======================================================================================
# Molecule 1 model
# =======================================================================================
fdir = '/Users/kimsj/Documents/RADMC-3D/radmc3d-2.0/RU_Lup_test/Automatics/Fin_script/fiducial_wind/'
fitsname = 'RULup_'+mole1+'_'+tname+'_'+b_maj+'.fits'  #
hdu = fits.open(fdir+fitsname)[0]   # Read the fits file: header + data
data = hdu.data#[0,0,:,:]      # Save the data part
nx = hdu.header['NAXIS1']; ny = hdu.header['NAXIS2']; nv = hdu.header['NAXIS3']  # Set up axis lengths
#if len(data.shape) == 3: data = data.reshape(-1,nv,ny,nx)    # Match the data shape to (1,nv, ny, nx)
#data = np.swapaxes(data, 0, 2); data = np.swapaxes(data, 1, 2) # reshape the data in (nx,ny,nv,1)
#print(data.max(),data.min())
dv = hdu.header['CDELT3'] ; v_ref = hdu.header['CRVAL3']  # delta_V in km/s
velax = np.arange(nv)*dv + v_ref
# Set continuum level
data_c = np.zeros((ny,nx))
for i in range(10):
    data_c += data[i,:,:]
data_c /= 10.
# Momemnt map calculations
M0, dM0 = collapse_zeroth(velax, data, 0, 0, nv)
M1, dM1 = collapse_first(velax, data, 1e-6, 3, 0, nv)
M2, dM2 = collapse_second(velax, data, 1e-6, 3, 0, nv)

# =======================================================================================
# Molecule 2 model
# =======================================================================================
fdir = '/Users/kimsj/Documents/RADMC-3D/radmc3d-2.0/RU_Lup_test/Automatics/Fin_script/fiducial_wind/'
fitsname = 'RULup_'+mole2+'_'+tname+'_'+b_maj+'.fits'  #
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
M1w, dM1w = collapse_first(velaxw, dataw, 1e-6, 3, 0, nvw)
M2w, dM2w = collapse_second(velaxw, dataw, 1e-6, 3, 0, nvw)

# =======================================================================================
# mole1 - mole2 of M1 and M2 maps
# =======================================================================================
Del_M1 = M1 - M1w
Del_M2 = M2 - M2w

# Set additional contours & Beam size
wcs = WCS(hdu.header).slice([nx,ny])                         # calculate RA, Dec
bmaj=hdu.header['BMAJ']; bmin=hdu.header['BMIN']; bpa=hdu.header['BPA'] # Read beam parameter
bmaj=(bmaj*u.degree).to(u.arcsec).value; bmin=(bmin*u.degree).to(u.arcsec).value # Convert from degree to arcsec
cell=abs(hdu.header['CDELT2']); Cell=(cell*u.degree).to(u.arcsec).value # Read pixel size in arcsec
beam = Ellipse((50., 50.), bmaj*100, bmin*100, angle=90.+bpa, edgecolor='white', facecolor='black')
disk = Ellipse((250,250), 240./150.*100, 240./150.*100*np.cos(inc*np.pi/180.), angle=31., edgecolor='white', facecolor='none',ls='--')  # Keplerian disk
ring = Ellipse((250,250), 520./150.*100, 520./150.*100*np.cos(inc*np.pi/180.), angle=31., edgecolor='cyan', facecolor='none',ls='--')   # Outer envelope boundary
# Moment 1 map -----------------------------------------------
fig=plt.figure(num=None, figsize=(8,6), dpi=100, facecolor='w', edgecolor='k')
ax = fig.add_subplot(projection=wcs)
im=ax.imshow(Del_M1,origin='lower',vmax=0.5,vmin=-0.5,cmap="jet")
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
plt.savefig('./moments_maps/RULup_'+mole1+'-'+mole2+'_'+tname+'_'+b_maj+'_M1.pdf', bbox_inches='tight', pad_inches=0.1,dpi=100)
#plt.show()
plt.clf()

# Set additional contours & Beam size
beam = Ellipse((50., 50.), bmaj*100, bmin*100, angle=90.+bpa, edgecolor='white', facecolor='black')
disk = Ellipse((250,250), 240./150.*100, 240./150.*100*np.cos(inc*np.pi/180.), angle=31., edgecolor='white', facecolor='none',ls='--')  # Keplerian disk
ring = Ellipse((250,250), 520./150.*100, 520./150.*100*np.cos(inc*np.pi/180.), angle=31., edgecolor='cyan', facecolor='none',ls='--')   # Outer envelope boundary
# Moment 2 map -----------------------------------------------
fig=plt.figure(num=None, figsize=(8,6), dpi=100, facecolor='w', edgecolor='k')
ax = fig.add_subplot(projection=wcs)
im=ax.imshow(Del_M2,origin='lower',vmax=0.5,vmin=-0.5,cmap="jet")
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
plt.savefig('./moments_maps/RULup_'+mole1+'-'+mole2+'_'+tname+'_'+b_maj+'_M2.pdf', bbox_inches='tight', pad_inches=0.1,dpi=100)
#plt.show()
plt.clf()
