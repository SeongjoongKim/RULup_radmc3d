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
parser.add_argument('-bmaj', default='bmaj5', type=str, help='The beam size. Default is bmaj5.')
parser.add_argument('-tname', default='fiducial', type=str, help='Test name. Default is fiducial.')
args = parser.parse_args()

mole = args.mole  #'CN_3-2'
b_maj = args.bmaj  #'bmaj51'
tname = args.tname  #'fiducial'

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
fdir = '/Users/kimsj/Documents/RADMC-3D/radmc3d-2.0/RU_Lup_test/Automatics/Fin_script/fiducial/'
fitsname = 'RULup_'+mole+'_'+tname+'_'+b_maj+'.fits'  #
if not os.path.exists(fdir+fitsname):
    fitsname = 'RULup_'+mole+'_fiducial_'+b_maj+'.fits'
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
M1, dM1 = collapse_first(velax, data-data_c, 1e-6, 3, 0, nv)
M2, dM2 = collapse_second(velax, data-data_c, 1e-6, 3, 0, nv)

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
im=ax.imshow(M1,origin='lower',vmax=0.5,vmin=-0.5,cmap="jet")
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
plt.savefig('./moments_maps/RULup_'+mole+'_'+tname+'_'+b_maj+'_M1.pdf', bbox_inches='tight', pad_inches=0.1,dpi=100)
#plt.show()
plt.clf()

# Set additional contours & Beam size
beam = Ellipse((50., 50.), bmaj*100, bmin*100, angle=90.+bpa, edgecolor='white', facecolor='black')
disk = Ellipse((250,250), 240./150.*100, 240./150.*100*np.cos(inc*np.pi/180.), angle=31., edgecolor='white', facecolor='none',ls='--')  # Keplerian disk
ring = Ellipse((250,250), 520./150.*100, 520./150.*100*np.cos(inc*np.pi/180.), angle=31., edgecolor='cyan', facecolor='none',ls='--')   # Outer envelope boundary
# Moment 2 map -----------------------------------------------
fig=plt.figure(num=None, figsize=(8,6), dpi=100, facecolor='w', edgecolor='k')
ax = fig.add_subplot(projection=wcs)
im=ax.imshow(M2,origin='lower',vmax=1.5,vmin=0.0,cmap="gist_heat")
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
plt.savefig('./moments_maps/RULup_'+mole+'_'+tname+'_'+b_maj+'_M2.pdf', bbox_inches='tight', pad_inches=0.1,dpi=100)
#plt.show()
plt.clf()


# =======================================================================================
# wind model
# =======================================================================================
fdir = '/Users/kimsj/Documents/RADMC-3D/radmc3d-2.0/RU_Lup_test/Automatics/Fin_script/fiducial_wind/'
if len(tname.split('_')) == 1:
    tname_wind = tname+'_wind'
else:
    tname_wind = tname #tname.split('_')[0]+'_wind_'+tname.split('_')[1]
fitsname = 'RULup_'+mole+'_'+tname_wind+'_'+b_maj+'.fits'  #
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

# Set additional contours & Beam size
wcsw = WCS(hduw.header).slice([nxw,nyw])                         # calculate RA, Dec
bmajw=hduw.header['BMAJ']; bminw=hduw.header['BMIN']; bpaw=hduw.header['BPA'] # Read beam parameter
bmajw=(bmajw*u.degree).to(u.arcsec).value; bminw=(bminw*u.degree).to(u.arcsec).value # Convert from degree to arcsec
cellw=abs(hduw.header['CDELT2']); Cellw=(cellw*u.degree).to(u.arcsec).value # Read pixel size in arcsec
beam = Ellipse((50., 50.), bmajw*100, bminw*100, angle=90.+bpaw, edgecolor='white', facecolor='black')
disk = Ellipse((250,250), 240./150.*100, 240./150.*100*np.cos(inc*np.pi/180.), angle=31., edgecolor='white', facecolor='none',ls='--')  # Keplerian disk
ring = Ellipse((250,250), 520./150.*100, 520./150.*100*np.cos(inc*np.pi/180.), angle=31., edgecolor='black', facecolor='none',ls='--')   # Outer envelope boundary
# Moment 1 map -----------------------------------------------
fig=plt.figure(num=None, figsize=(8,6), dpi=100, facecolor='w', edgecolor='k')
ax = fig.add_subplot(projection=wcsw)
im=ax.imshow(M1w,origin='lower',vmax=0.5,vmin=-0.5,cmap="jet")
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
plt.savefig('./moments_maps/RULup_'+mole+'_'+tname_wind+'_'+b_maj+'_M1.pdf', bbox_inches='tight', pad_inches=0.1,dpi=100)
#plt.show()
plt.clf()

# Set additional contours & Beam size
beam = Ellipse((50., 50.), bmajw*100, bminw*100, angle=90.+bpaw, edgecolor='white', facecolor='black')
disk = Ellipse((250,250), 240./150.*100, 240./150.*100*np.cos(inc*np.pi/180.), angle=31., edgecolor='white', facecolor='none',ls='--')  # Keplerian disk
ring = Ellipse((250,250), 520./150.*100, 520./150.*100*np.cos(inc*np.pi/180.), angle=31., edgecolor='cyan', facecolor='none',ls='--')   # Outer envelope boundary
# Moment 2 map -----------------------------------------------
fig=plt.figure(num=None, figsize=(8,6), dpi=100, facecolor='w', edgecolor='k')
ax = fig.add_subplot(projection=wcs)
im=ax.imshow(M2w,origin='lower',vmax=1.5,vmin=0.0,cmap="gist_heat")
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
plt.savefig('./moments_maps/RULup_'+mole+'_'+tname_wind+'_'+b_maj+'_M2.pdf', bbox_inches='tight', pad_inches=0.1,dpi=100)
#plt.show()
plt.clf()

#'''
# =======================================================================================
# Wind - No wind M1 and M2 maps
# =======================================================================================
Del_M1 = M1w - M1
Del_M2 = M2w - M2

# Set additional contours & Beam size
wcsw = WCS(hduw.header).slice([nxw,nyw])                         # calculate RA, Dec
bmajw=hduw.header['BMAJ']; bminw=hduw.header['BMIN']; bpaw=hduw.header['BPA'] # Read beam parameter
bmajw=(bmajw*u.degree).to(u.arcsec).value; bminw=(bminw*u.degree).to(u.arcsec).value # Convert from degree to arcsec
cellw=abs(hduw.header['CDELT2']); Cellw=(cellw*u.degree).to(u.arcsec).value # Read pixel size in arcsec
beam = Ellipse((50., 50.), bmajw*100, bminw*100, angle=90.+bpaw, edgecolor='white', facecolor='black')
disk = Ellipse((250,250), 240./150.*100, 240./150.*100*np.cos(inc*np.pi/180.), angle=31., edgecolor='white', facecolor='none',ls='--')  # Keplerian disk
ring = Ellipse((250,250), 520./150.*100, 520./150.*100*np.cos(inc*np.pi/180.), angle=31., edgecolor='black', facecolor='none',ls='--')   # Outer envelope boundary
# Moment 1 map -----------------------------------------------
fig=plt.figure(num=None, figsize=(8,6), dpi=100, facecolor='w', edgecolor='k')
ax = fig.add_subplot(projection=wcsw)
im=ax.imshow(Del_M1,origin='lower',vmax=0.25,vmin=-0.25,cmap="jet")
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
plt.savefig('./moments_maps/RULup_'+mole+'_'+tname_wind+'_'+b_maj+'_DEL_M1.pdf', bbox_inches='tight', pad_inches=0.1,dpi=100)
#plt.show()
plt.clf()

# Set additional contours & Beam size
beam = Ellipse((50., 50.), bmajw*100, bminw*100, angle=90.+bpaw, edgecolor='white', facecolor='black')
disk = Ellipse((250,250), 240./150.*100, 240./150.*100*np.cos(inc*np.pi/180.), angle=31., edgecolor='white', facecolor='none',ls='--')  # Keplerian disk
ring = Ellipse((250,250), 520./150.*100, 520./150.*100*np.cos(inc*np.pi/180.), angle=31., edgecolor='black', facecolor='none',ls='--')   # Outer envelope boundary
# Moment 2 map -----------------------------------------------
fig=plt.figure(num=None, figsize=(8,6), dpi=100, facecolor='w', edgecolor='k')
ax = fig.add_subplot(projection=wcs)
im=ax.imshow(Del_M2,origin='lower',vmax=0.25,vmin=-0.25,cmap="jet")
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
plt.savefig('./moments_maps/RULup_'+mole+'_'+tname_wind+'_'+b_maj+'_DEL_M2.pdf', bbox_inches='tight', pad_inches=0.1,dpi=100)
#plt.show()
plt.clf()
#'''


# =======================================================================================
# M2 azimuthal average profiles
# =======================================================================================
# Radius range set
r_min = 0.0;  r_max = 160.0   # in au unit
dr = 8.0; nr = int(r_max/dr)
r_bin = np.arange(r_min,r_max+dr,dr)
rc = (r_bin[1:nr+1] + r_bin[0:nr])*0.5
d_pc = 160.0 # Distant of the source
outname = './moments_maps/RULup_'+mole+'_'+tname_wind+'_'+b_maj+'_M2_aver_radial.dat'
# M2 azimuthal average profile
M2_aver = np.zeros_like(rc)
M2w_aver = np.zeros_like(rc)
cube = imagecube(fdir + fitsname)
rvals, tvals, _ = cube.disk_coords(x0=dxc,y0=dyc,inc=inc,PA=DPA,z0=z0,psi=psi,z1=z1,phi=phi)
r_au = rvals * d_pc
for j in range(nr):
    r_mask = np.logical_and(r_au >= r_bin[j], r_au <= r_bin[j+1])
    PA_mask = np.logical_and(tvals >= PA_min*np.pi/180., tvals <= PA_max*np.pi/180.)
    #v_mask = np.logical_and(vmodel >= -5.0e3, vmodel <= 5.0e3)
    mask = r_mask*PA_mask#*v_mask
    n_sample = np.sum(mask) ; idx,idy = np.where(mask >0)
    M2_sample = np.zeros(n_sample); M2w_sample = np.zeros(n_sample)
    for k in range(len(idx)):
        M2_sample[k] = M2[idx[k],idy[k]]
        M2w_sample[k] = M2w[idx[k],idy[k]]
    M2_aver[j] = M2_sample.mean(); M2w_aver[j] = M2w_sample.mean()
outfile=open(outname, 'w+')
data=np.array((rc,M2_aver,M2w_aver))
data=data.T
np.savetxt(outfile, data, fmt='%13.8f', delimiter='  ',newline='\n')
outfile.close()
