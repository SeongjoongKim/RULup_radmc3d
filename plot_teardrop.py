import numpy as np
from Model_setup_subroutines import *
#from matplotlib.colors import *
import matplotlib.colors as colors
import matplotlib.pyplot as plt
from matplotlib.offsetbox import (AnchoredOffsetbox, AuxTransformBox, DrawingArea, TextArea, VPacker)
from astropy.io import fits
from astropy.wcs import WCS
from scipy.interpolate import interp1d, griddata
from scipy.stats import binned_statistic
from astropy.convolution import Gaussian2DKernel, convolve
from matplotlib.patches import Ellipse, Rectangle, Circle
import argparse, os
from disksurf import observation
from gofish import imagecube

# Some natural constants  ----------------------------------
au  = 1.49598e13     # Astronomical Unit       [cm]
pc  = 3.08572e18     # Parsec                  [cm]
ms  = 1.98892e33     # Solar mass              [g]
ts  = 5.78e3         # Solar temperature       [K]
ls  = 3.8525e33      # Solar luminosity        [erg/s]
rsun  = 6.96e10        # Solar radius            [cm]
GG  = 6.67408e-08    # Gravitational constant  [cm^3/g/s^2]
mp  = 1.6726e-24     # Mass of proton          [g]
SB = 5.670374419e-5    # Stefan-Boltzmann constant [ erg cm-2 s-1 K-4]
#      Physical constants ----------------------------------
mu = 2.34
kb = 1.38e-16
NA = 6.02e23
mH = 1.0/NA

# =======================================================================================
# Functions
# =======================================================================================
def rotated_axis(inc, PA):
    aa = np.sin(inc*np.pi/180.); bb = np.cos(inc*np.pi/180.)
    cc = np.sin(DPA*np.pi/180.); dd = np.cos(DPA*np.pi/180.)
    # From the projection
    # inclination: +z moves to +y axis
    # DPA: +x moves to +y axis
    ux_sky = np.array (( dd, cc*bb, -cc*aa ))
    uy_sky = np.array (( -cc, dd*bb, -dd*aa ))
    uz_sky = np.array (( 0, aa, bb ))
    # From the conversion between RA-DEC and Azi-Alt
    # inclination: +z moves to +y axis
    # DPA: +x moves to +y axis
    #lamx = np.arctan2(cc*bb,dd)
    #betax = np.arcsin(-aa*cc)
    #lamy = np.arctan2(dd*bb,-cc)
    #betay = np.arcsin(-aa*dd)
    #ux_sky = np.array (( np.cos(lamx)*np.cos(betax), np.sin(lamx)*np.cos(betax), np.sin(betax) ))
    #uy_sky = np.array (( np.cos(lamy)*np.cos(betay), np.sin(lamy)*np.cos(betay), np.sin(betay) ))
    #uz_sky = np.array (( ux_sky[1]*uy_sky[2]-ux_sky[2]*uy_sky[1], ux_sky[2]*uy_sky[0]-ux_sky[0]*uy_sky[2], ux_sky[0]*uy_sky[1]-ux_sky[1]*uy_sky[0] ))
    #print(ux_sky, uy_sky, uz_sky)
    return ux_sky, uy_sky, uz_sky


def vector_proj(x,y,z,u_sky):
    return x*u_sky[0] + y*u_sky[1] + z*u_sky[2]
    

def get_mask(r,t,rin,rout,PAmin,PAmax):
    r_mask= np.logical_and(r>=rin, r<=rout)
    t_mask = np.logical_and(t>=np.radians(PAmin),t<=np.radians(PAmax))
    mask = r_mask*t_mask#; npnts = np.sum(mask)
    return mask
    
    
def get_vlos_spec(vmodel, cube, mask):
    npnts = np.sum(mask); k=0
    vl = np.zeros(npnts); inten = np.zeros((npnts,cube.shape[0]))
    for i in range(cube.shape[2]):
        for j in range(cube.shape[1]):
            if mask[i,j]:
                vl[k] = vmodel[i,j]
                inten[k,:] = cube[:,j,i]
                k += 1
    return vl, inten

def shift_spec(data, vax, rr, tt, vmodel, r_min, r_max, PA_min, PA_max):
    r_mask = np.logical_and(rr >= r_min, rr <= r_max)
    PA_mask = np.logical_and(tt >= PA_min*np.pi/180., tt <= PA_max*np.pi/180.)
    #v_mask = np.logical_and(vmodel >= -5.0e3, vmodel <= 5.0e3)
    mask = r_mask*PA_mask#*v_mask
    nv, ny, nx = data.shape
    ndpnts = np.sum(mask)
    shifted = np.zeros(nv)
    ndata = np.ones(nv)*ndpnts
    for i in range(nx):
        for j in range(ny):
            if mask[j,i]:
                shifted_spec = interp1d(vax - vmodel[i, j], data[:, j, i], bounds_error=False)(vax)
                ndata[np.isnan(shifted_spec)] -= 1
                shifted_spec[np.isnan(shifted_spec)] = 0.0
                shifted += shifted_spec
    shifted /= ndata
    return shifted


def azi_aver(data, rr, tt, r_min, r_max, PA_min, PA_max):
    r_mask = np.logical_and(rr >= r_min, rr <= r_max)
    PA_mask = np.logical_and(tt >= PA_min*np.pi/180., tt <= PA_max*np.pi/180.)
    mask = r_mask*PA_mask
    n_sample = np.sum(mask) ; idx,idy = np.where(mask >0)
    for k in range(len(idx)):
        I_sample = data[idx[k],idy[k]]
    I_avg = I_sample.mean(); I_std = np.std(I_sample)
    return I_avg, I_std
    
    
# =======================================================================================
# Parameter setup
# =======================================================================================
parser = argparse.ArgumentParser()
parser.add_argument('-mole', default='None', type=str, help='The molecular line name. Default is None.')
parser.add_argument('-wc', default='F', type=str, help='If you want to include continuum, set this argument T. default is F (no continuum).')
args = parser.parse_args()
mole =  args.mole  # '13CO_3-2' #

# =======================================================================================
# Disk parameters
# =======================================================================================
mstar, rstar, tstar = [ 0.63*ms, 2.42*rsun, 4073. ] #Stellar parameters. The list has [M_star, R_star, T_star] in Solar mass, Solar radius, and Kelvin unit,
inc = 25.0
DPA = 149.0; PA_min = -180.0; PA_max = 180.0
dxc = 0.00; dyc = 0.00
z0 = 0.3; psi = 1.4; z1 = -0.0; phi = 1.0
#r_taper = 260.; q_taper = 1.0
d_pc = 160.0 # Distant of the source in pc
vsys = 4.5   # System velocity in km/s

# =======================================================================================
# Read observational data
# =======================================================================================
# Set fits file # =======================================================================
fdir = '/Users/kimsj/Documents/RU_Lup/Fin_fits/'
fitsname = mole+'_selfcal_wc_matched_cube500.fits'  #
hdu = fits.open(fdir+fitsname)[0]   # Read the fits file: header + data
# Read header # =========================================================================
nx = hdu.header['NAXIS1']; ny = hdu.header['NAXIS2']; nv = hdu.header['NAXIS3']  # Set up axis lengths
bmaj = hdu.header['BMAJ']*3.6e3; bmin = hdu.header['BMIN']*3.6e3; bpa = hdu.header['BPA']
pixsize_x = abs(hdu.header['CDELT1']*3.6e3); pixsize_y = abs(hdu.header['CDELT2']*3.6e3)
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

# Set velocity axis of the data cube
velax = np.arange(nv)*dv + v0
if hdu.header['NAXIS'] == 4: data = hdu.data[0,:,:,:]      # Save the data part
if hdu.header['NAXIS'] == 3: data = hdu.data#[0,:,:,:]      # Save the data part

# =======================================================================================
# vector projection methods
# =======================================================================================
# Axes rotation from disk plane to sky plane
# inclination: +z moves to +y axis
# DPA: +x moves to +y axis
# The disk inclined and rotated inversely. +y axis becomes near side & disk major axis rotates PA toward -y axis.
ux_sky, uy_sky, uz_sky = rotated_axis(inc, DPA)
print(ux_sky, uy_sky, uz_sky)
# Axes rotation from sky plane to disk plane
ux_disk, uy_disk, uz_disk = np.linalg.inv(np.array((ux_sky,uy_sky,uz_sky)))
print(ux_disk, uy_disk, uz_disk)

# Set sky plane xy # =========================================================
x=np.arange(-nx/2.,nx/2.)*pixsize_x#*d_pc # in au
y=np.arange(-ny/2.,ny/2.)*pixsize_y#*d_pc # in au
qq = np.meshgrid(x,y)
x_sky,y_sky = np.ravel(qq[0]), np.ravel(qq[1])

# Set disk coordinate vector in disk # ==========================================
x_disk, y_disk = 2.*x_sky, 2.*y_sky
r_disk = np.hypot(x_disk,y_disk)
z_disk = ( z0 * np.power(r_disk, psi) + z1 * np.power(r_disk, phi) ) #*np.exp(-np.power(r_disk / 750., 1.0)) # z(r) = z0(r/1'')**psi - z1(r/1'')**phi
# Rotation velocity vector in disk planet
#vkep_disk = np.sqrt(GG*mstar/np.hypot(r,z_disk)/au)*1e-5
#vkep_disk[np.isnan(vkep_disk)] = 0.0
#theta = np.arctan2(y_disk,x_disk)
#vx_disk = vkep_disk*np.sin(theta); vy_disk = vkep_disk*np.cos(theta); vz_disk = np.zeros_like(vx_disk)

# Calculate the projected (x', y', z') & (vx', vy', vz') in sky plane # ==============================
xx_sky = vector_proj(x_disk, y_disk, z_disk, ux_sky)
yy_sky = vector_proj(x_disk, y_disk, z_disk, uy_sky)
zz_sky = vector_proj(x_disk, y_disk, z_disk, uz_sky)
#vvz_sky = vector_proj(vx_disk, vy_disk, vz_disk, uz_sky)
#rr = np.hypot(xx,yy)

# Interpolate z_prj & v_kep on the sky plane # =======================================================
#z_prj = griddata((xx,yy),z_disk,(x_sky,y_sky),method='linear')
#vrot = griddata((xx,yy),vvz_sky,(x_sky,y_sky),method='linear')

# Interpolate z_sky # ================================================================================
z_sky = griddata((xx_sky,yy_sky),zz_sky,(x_sky,y_sky),method='linear')
# Deproject observed pixels on the disk plane # ======================================================
xx_disk = vector_proj(x_sky, y_sky, z_sky, ux_disk)
yy_disk = vector_proj(x_sky, y_sky, z_sky, uy_disk)
zz_disk = vector_proj(x_sky, y_sky, z_sky, uz_disk)
# Rotation velocity vector in the deprojected disk planet # ==========================================
rr_disk = np.hypot(xx_disk,yy_disk)
vvkep_disk = 1e-5*np.sqrt(GG*mstar/np.hypot(rr_disk,zz_disk)/d_pc/au)  # Vkep = sqrt(G*Mstar/d)
#vvkep_disk = GG*mstar*np.power(rr_disk*d_pc*au, 2.0)   # Vkep = sqrt(G*Mstar*r2/d3)
#vvkep = np.sqrt( vvkep/np.power( np.hypot(rr_disk,zz_disk)*au*d_pc, 3.0) ) *1e-5 # Keplerian velocity in disk coordinates with km/s unit
vvkep_disk = np.where(np.isfinite(vvkep_disk), vvkep_disk, 0.0)       # Check Nan and +/-inf and replace them to 0
ttheta = np.arctan2(yy_disk,xx_disk)
vvx_disk = vvkep_disk*np.sin(ttheta); vvy_disk = vvkep_disk*np.cos(ttheta); vvz_disk = np.zeros_like(vvx_disk)
vvz_sky = vector_proj(vvx_disk, vvy_disk, vvz_disk, uz_sky) # Final velocity field on the sky plane
vvz_sky = np.where(np.isfinite(vvz_sky), vvz_sky, 0.0)      # Check Nan and +/-inf and replace them to 0

# Convolution of V field by a beam size
gaussian_2D_kernel = Gaussian2DKernel(bmaj/pixsize_x/np.sqrt(8*np.log(2)),bmin/pixsize_y/np.sqrt(8*np.log(2)),bpa/180.*np.pi,x_size=151, y_size=151)
# Convolve the image through the 2D beam
vmod = convolve(vvz_sky.reshape((nx,ny)),gaussian_2D_kernel,boundary='extend',normalize_kernel=True)
'''
# Plotting the rotation field # =======================================================================
wcs = WCS(hdu.header).slice([nx,ny])                         # calculate RA, Dec
#disk = Ellipse((250,250), 240./150.*100, 240./150.*100*np.cos(inc*np.pi/180.), angle=31., edgecolor='white', facecolor='none',ls='--')  # Keplerian disk
#ring = Ellipse((250,250), 520./150.*100, 520./150.*100*np.cos(inc*np.pi/180.), angle=31., edgecolor='black', facecolor='none',ls='--')   # Outer envelope boundary
fig=plt.figure(num=None, figsize=(8,6), dpi=100, facecolor='w', edgecolor='k')
ax = fig.add_subplot(projection=wcs)
im=ax.imshow(vmod,origin='lower',cmap="jet",vmax=1.0,vmin=-1.0)
TT = ax.contour(zz_disk.reshape((nx,ny)),[0.1,0.5,1.0,3.0],linestyles='solid',colors='gray')
plt.clabel(TT)
ax.set_xlabel('RA',fontsize=15)
ax.set_ylabel('Dec',fontsize=15)
#ax.add_patch(disk)
#ax.add_patch(ring)
#ax.text(0.95, 0.95, freq[i], ha='right', va='top', transform=ax.transAxes, color="w",fontsize=15)
ax.tick_params(axis='both',which='both',length=4,width=1,labelsize=12,direction='in',color='w')
#ax.margins(x=-0.375,y=-0.375)
cbar=plt.colorbar(im, shrink=0.9)
cbar.set_label(r'$\mathrm{(km\ s^{-1})^{2}}$', size=10)
plt.savefig('./teardrop/Flared_emitting_layer_vkep_vector.pdf', bbox_inches='tight', pad_inches=0.1,dpi=100)
#plt.show()
plt.clf()
'''
#'''
# =======================================================================================
# Making teardrop plot
# =======================================================================================
# Radius range set # ====================================================================
r_min = 0.0;  r_max = 2.5; dr = 0.05   # in au arcsec
nr = int(r_max/dr)
r_bin = np.arange(r_min,r_max+dr,dr)
rc = (r_bin[1:nr+1] + r_bin[0:nr])*0.5

# Set continuum level # =================================================================
if args.wc == 'F':
    data_cl = np.zeros((ny,nx))
    data_cr = np.zeros((ny,nx))
    for i in range(20):
        data_cl += data[i,:,:]
        data_cr += data[nv-1-i,:,:]
    data_cl /= 20.; data_cr /= 20.
    data_c = (data_cl+data_cr)*0.5
    data -= data_c[None,:,:]  # continuum subtraction

# Set vel axis # ========================================================================
vref = (np.arange(nv+1)-0.5)*dv + v0
vcent = np.average([vref[1:], vref[:-1]], axis=0)
aver_spec = np.zeros((nr,nv)); std_spec = np.zeros((nr,nv))
# calculate averaged spectra # ==========================================================
for i in range(len(rc)):
    #print(i)
    r_in = r_bin[i]; r_out = r_bin[i+1]
    mask = get_mask(rr_disk.reshape((nx,ny)),ttheta.reshape((nx,ny)),r_in,r_out,PA_min,PA_max)
    vlos, spectrum = get_vlos_spec(vmod, data, mask)
    vpnts = np.ravel(velax[None,:]-vlos[:,None]); spnts = np.ravel(spectrum)
    id = np.argsort(vpnts)
    vsort = vpnts[id]; ssort = spnts[id];
    y = binned_statistic(vsort, ssort, statistic='mean', bins=vref)[0]
    aver_spec[i,:] = np.where(np.isfinite(y), y, 0.0)
    std_spec[i,:] = binned_statistic(vsort, ssort, statistic='std', bins=vref)[0]
    #spectra[i,:] = shift_spec(datao[0,:,:,:], velax, rr_disk.reshape((nx,ny)), ttheta.reshape((nx,ny)), vmod+vsys, r_min, r_max, PA_min, PA_max)

# Plot teardrop figure # ================================================================
plt.figure(figsize=(10,6))
plt.pcolormesh(vcent,rc,aver_spec,norm=colors.LogNorm(vmin=1e-2*aver_spec.max(),vmax=aver_spec.max()),cmap='gist_heat',shading='gouraud',rasterized=True)
plt.axvline(x=5.0,color='m',ls='--')
plt.axvline(x=4.5,color='g',ls='--')
plt.xlim(-2,10)
#plt.ylim(0.0,2.5)
plt.xlabel(r'V$_{radio}$ (km/s)', fontsize=15)
plt.ylabel('R (arcsec)',fontsize=15)
#plt.legend(prop={'size':15},loc=1)
plt.tick_params(which='both',length=6,width=1.5,direction='in',color='w')
cbar = plt.colorbar() #ticks=[1e4,1e5,1e6,1e7])
cbar.set_label(r'$Jy\ beam^{-1}$', size=10)
plt.savefig('./teardrop/'+mole+'_teardrop_vector.pdf', bbox_inches='tight', pad_inches=0.1,dpi=100)
plt.clf()
#'''
'''
# =======================================================================================
# Testing GoFish teardrop plot
# =======================================================================================
cube = imagecube(fdir+fitsname)
r_asc, velax, spectra, scatter  = cube.radial_spectra(inc=inc,PA=DPA,mstar=0.64,dist=160.0,dr=0.05,z0=z0,psi=psi,z1=z1,phi=phi)
plt.figure(figsize=(10,6))
plt.pcolormesh(1e-3*velax,r_asc,spectra,norm=colors.LogNorm(vmin=1e-2*spectra.max(),vmax=spectra.max()),cmap='gist_heat',shading='gouraud',rasterized=True)
#plt.xlim(-1,2.4)
#plt.ylim(-2,-0.25)
plt.xlabel(r'V$_{radio}$ (km/s)', fontsize=15)
plt.ylabel('R (arcsec)',fontsize=15)
#plt.legend(prop={'size':15},loc=1)
plt.tick_params(which='both',length=6,width=1.5)
cbar = plt.colorbar() #ticks=[1e4,1e5,1e6,1e7])
cbar.set_label(r'$Jy\ beam^{-1}$', size=10)
plt.savefig('./'+mole+'_teardrop.pdf', bbox_inches='tight', pad_inches=0.1,dpi=100)
'''
'''
# =======================================================================================
# Finding emitting surface by disksurf (Teague et al. 2021; Pinte et al. 2018)
# =======================================================================================
mole = '12CO_2-1'
fdir = '/Users/kimsj/Documents/RU_Lup/Fin_fits/'
fitsname = mole+'_selfcal_matched_cube500.fits'
hdu = fits.open(fdir+fitsname)[0]   # Read the fits file: header + data
nx = hdu.header['NAXIS1']; ny = hdu.header['NAXIS2']; nv = hdu.header['NAXIS3']  # Set up axis lengths
bmaj = hdu.header['BMAJ']*3.6e3; bmin = hdu.header['BMIN']*3.6e3; bpa = hdu.header['BPA']
pixsize_x = abs(hdu.header['CDELT1']*3.6e3); pixsize_y = abs(hdu.header['CDELT2']*3.6e3)
cube = observation(fdir+fitsname)
chans = (50,105) # (50,105) for 2-1 lines / (30,70) for C18O 3-2 / (80,150) for 13CO 3-2 & CN 3-2
#cube.plot_channels(chans=chans)
surface = cube.get_emission_surface(inc=inc,PA=DPA,r_min=0.1,r_max=1.5,chans=chans,smooth=0.5)
rf_surf, zf_surf = [surface.r(side='front'), surface.z(side='front')]
rb_surf, zb_surf = [surface.r(side='back'), surface.z(side='back')]
plt.figure(figsize=(10,6))
plt.scatter(rf_surf, zf_surf,marker='o',color='blue')
plt.scatter(rb_surf, zb_surf,marker='o',color='red')
plt.xlim(0,1.5)
plt.ylim(-0.4,0.4)
plt.xlabel('R (arcsec)', fontsize=15)
plt.ylabel('Z (arcsec)',fontsize=15)
#plt.legend(prop={'size':15},loc=1)
plt.tick_params(which='both',length=6,width=1.5,direction='in')
#surface.plot_surface()
plt.savefig('./teardrop/'+mole+'_obs_disksurf_zsurface.pdf',dpi=100,bbox_inches='tight')
plt.clf()

chans = (70,110) # (70,105) #
cube.plot_peaks(surface=surface)
plt.savefig('./teardrop/'+mole+'_obs_disksurf_peaks.pdf',dpi=100,bbox_inches='tight')
'''
'''
# =======================================================================================
# Finding emitting surface by RADMC-3D tausurf
# =======================================================================================
mole = 'CN_3-2'
fdir = '/Users/kimsj/Documents/RADMC-3D/radmc3d-2.0/RU_Lup_test/Automatics/Fin_script/tau1_surf_fits/'
fitsname = 'RULup_'+mole+'_fiducial_wind_all_bmaj51_tau1.fits'
hdu = fits.open(fdir+fitsname)[0]   # Read the fits file: header + data
nx = hdu.header['NAXIS1']; ny = hdu.header['NAXIS2']; nv = hdu.header['NAXIS3']  # Set up axis lengths
bmaj = hdu.header['BMAJ']*3.6e3; bmin = hdu.header['BMIN']*3.6e3; bpa = hdu.header['BPA']
pixsize_x = abs(hdu.header['CDELT1']*3.6e3); pixsize_y = abs(hdu.header['CDELT2']*3.6e3)
if hdu.header['NAXIS'] == 4: data = hdu.data[0,:,:,:]      # Save the data part
if hdu.header['NAXIS'] == 3: data = hdu.data#[0,:,:,:]      # Save the data part
# Find the z_peak of tau = 1 surface
z_sky = np.zeros((nx,ny))
for i in range(nx):
    for j in range(ny):
        z_sky[i,j]=data[:,j,i].max()/au/d_pc
        if z_sky[i,j] == 0.0: z_sky[i,j] /= 0.0   # The 

# The disk inclined and rotated inversely. +y axis becomes near side & disk major axis rotates PA toward -y axis.
ux_sky, uy_sky, uz_sky = rotated_axis(inc, DPA)
print(ux_sky, uy_sky, uz_sky)
# Axes rotation from sky plane to disk plane
ux_disk, uy_disk, uz_disk = np.linalg.inv(np.array((ux_sky,uy_sky,uz_sky)))
print(ux_disk, uy_disk, uz_disk)
x=np.arange(-nx/2.,nx/2.)*pixsize_x#*d_pc # in au
y=np.arange(-ny/2.,ny/2.)*pixsize_y#*d_pc # in au
qq = np.meshgrid(x,y)
x_sky,y_sky = qq[0], qq[1]

xx_disk = vector_proj(x_sky, y_sky, z_sky, ux_disk)
yy_disk = vector_proj(x_sky, y_sky, z_sky, uy_disk)
zz_disk = vector_proj(x_sky, y_sky, z_sky, uz_disk)
zz_disk = np.where(np.isnan(zz_disk), 0.0, zz_disk)
rr_disk = np.hypot(xx_disk,yy_disk)

r = np.linspace(0.0,2.0,100)
z0 = 0.45; psi = 1.25
# Plotting the deprojected r-z points    
plt.figure(figsize=(10,6))
plt.scatter(rr_disk, zz_disk,marker='.',color='blue')
plt.plot(r, z0*np.power(r,psi),'r--',label='{:4.2f} r^{:4.2f}'.format(z0,psi))
plt.xlim(0,2.0)
#plt.ylim(0.0,0.5)
plt.xlabel('R (arcsec)', fontsize=15)
plt.ylabel('Z (arcsec)',fontsize=15)
plt.legend(prop={'size':15},loc=0)
plt.tick_params(which='both',length=6,width=1.5,direction='in')
plt.savefig('./teardrop/tausurf/'+mole+'_radmc3d_tausurf_zsurface.pdf',dpi=100,bbox_inches='tight')
#plt.show()
plt.clf()
'''

