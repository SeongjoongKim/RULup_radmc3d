from Model_setup_subroutines import *
from matplotlib.colors import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.gridspec as gridspec
import matplotlib.colors as colors
from astropy.io import fits
from astropy import units as u
from scipy.interpolate import interp1d, griddata
from astropy.convolution import Gaussian2DKernel, convolve
from matplotlib.patches import Ellipse, Rectangle, Circle
from astropy.wcs import WCS
from matplotlib.offsetbox import (AnchoredOffsetbox, AuxTransformBox, DrawingArea, TextArea, VPacker)#from Model_setup_subroutines import *
from gofish import imagecube
import argparse
import os

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
def coord_rotate(x,y,z,angle,axis=3):
    # Rotating the axes with angle against a given axis.
    # The rotation direction is clockwise:
    # when axis=3, +x axis rotates toward -y axis direction. (disk position angle)
    # when axis=1, +z axis rotates toward +y axis direction. (disk inclination)
    #x,y,z = xyz
    if axis==1:
        xx = x
        yy = y*np.cos(angle*np.pi/180.)-z*np.sin(angle*np.pi/180.)
        zz = y*np.sin(angle*np.pi/180.)+z*np.cos(angle*np.pi/180.)
    if axis==2:
        yy = y
        xx = x*np.cos(angle*np.pi/180.)-z*np.sin(angle*np.pi/180.)
        zz = x*np.sin(angle*np.pi/180.)+z*np.cos(angle*np.pi/180.)
    if axis==3:
        zz = z
        xx = x*np.cos(angle*np.pi/180.)-y*np.sin(angle*np.pi/180.)
        yy = x*np.sin(angle*np.pi/180.)+y*np.cos(angle*np.pi/180.)
    return xx,yy,zz


def rotated_axis(inc, PA):
    aa = np.sin(inc*np.pi/180.); bb = np.cos(inc*np.pi/180.)
    cc = np.sin(DPA*np.pi/180.); dd = np.cos(DPA*np.pi/180.)
    # From the projection
    # inclination: +z moves to -y axis
    # DPA: +x moves to +y axis
    #ux_sky = np.array (( dd, cc*bb, cc*aa ))
    #uy_sky = np.array (( -cc, dd*bb, dd*aa ))
    #uz_sky = np.array (( 0, -aa, bb ))
    # From the conversion between RA-DEC and Azi-Alt
    # inclination: +z moves to +y axis
    # DPA: +x moves to +y axis
    lamx = np.arctan2(cc*bb,dd)
    betax = np.arcsin(-aa*cc)
    lamy = np.arctan2(dd*bb,-cc)
    betay = np.arcsin(-aa*dd)
    ux_sky = np.array (( np.cos(lamx)*np.cos(betax), np.sin(lamx)*np.cos(betax), np.sin(betax) ))
    uy_sky = np.array (( np.cos(lamy)*np.cos(betay), np.sin(lamy)*np.cos(betay), np.sin(betay) ))
    uz_sky = np.array (( ux_sky[1]*uy_sky[2]-ux_sky[2]*uy_sky[1], ux_sky[2]*uy_sky[0]-ux_sky[0]*uy_sky[2], ux_sky[0]*uy_sky[1]-ux_sky[1]*uy_sky[0] ))
    #print(ux_sky, uy_sky, uz_sky)
    return ux_sky, uy_sky, uz_sky


def vector_proj(x,y,z,u_sky):
    xx = x*u_sky[0] + y*u_sky[1] + z*u_sky[2]
    return xx
    

# =======================================================================================
# Parameter setup
# =======================================================================================
parser = argparse.ArgumentParser()
parser.add_argument('-mole', default='None', type=str, help='The molecular line name. Default is None.')
#parser.add_argument('-wind', default='F', type=str, help='If you want to include wind model, set this argument T. default is F (no wind).')
args = parser.parse_args()
mole =  args.mole  # '13CO_3-2' #

# =======================================================================================
# Disk parameters & radii setup
# =======================================================================================
mstar, rstar, tstar = [ 0.63*ms, 2.42*rsun, 4073. ] #Stellar parameters. The list has [M_star, R_star, T_star] in Solar mass, Solar radius, and Kelvin unit,
inc = 25.0
DPA = 149.0; PA_min = -180.0; PA_max = 180.0
dxc = 0.00; dyc = 0.00
z0 = 0.5; psi = 1.0; z1 = -0.0; phi = 1.0
#r_taper = 260.; q_taper = 1.0
d_pc = 160.0 # Distant of the source

# Radius range set
r_min = 0.0;  r_max = 160.0   # in au unit
dr = 8.0; nr = int(r_max/dr)
r_bin = np.arange(r_min,r_max+dr,dr)
rc = (r_bin[1:nr+1] + r_bin[0:nr])*0.5
# =======================================================================================
# Read observational data
# =======================================================================================
fdir = '/Users/kimsj/Documents/RU_Lup/Fin_fits/'
fitsname = mole+'_selfcal_wc_matched_cube500.fits'  #
hdu = fits.open(fdir+fitsname)[0]   # Read the fits file: header + data
datao = hdu.data#[0,0,:,:]      # Save the data part
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
velaxo = np.arange(nv)*dv + v0
#'''
# =======================================================================================
# Setup the emitting surface
# =======================================================================================
# Setup the disk coordinates
x=np.arange(-nx/2.,nx/2.)*abs(hdu.header['CDELT1'])*3600.*d_pc/100. # in au
y=np.arange(-ny/2.,ny/2.)*abs(hdu.header['CDELT2'])*3600.*d_pc/100. # in au
qq = np.meshgrid(x,y)
x_sky,y_sky = np.ravel(qq[0]), np.ravel(qq[1])
x = 2.*x_sky; y = 2.*y_sky
r = np.hypot(x,y)
z = ( z0 * np.power(r, psi) + z1 * np.power(r, phi) )
#z *= np.exp(-np.power(r / r_taper, q_taper))

# Rotate inclination
xi,yi,zi = coord_rotate(x,y,z,-inc,axis=1)
r = np.hypot(xi,yi,zi)
vkep=np.sqrt(GG*mstar/r/au/d_pc)*np.sin(-inc*np.pi/180.)*np.cos(np.arctan2(y, x))
vkep[np.isnan(vkep)] = 0.0

# Rotate disk PA
xx,yy,zz = coord_rotate(xi,yi,zi,180.-DPA,axis=3)
#xx=xx.reshape((nx,ny)); yy=yy.reshape((nx,ny)); zz=zz.reshape((nx,ny))
#rr = np.sqrt(xx**2+yy**2+zz**2)
#vkep=np.sqrt(GG*mstar/rr/au)*np.sin(inc*np.pi/180.)*np.cos( np.arctan2(yy, xx) )
#vkep[np.isnan(vkep)] = 0.0

# interpolate vkep to the sky plane coordinates
vrot = griddata((xx,yy),vkep,(x_sky,y_sky),method='linear')
zz = griddata((xx,yy),z,(x_sky,y_sky),method='linear')
#vrot *= np.sin(inc*np.pi/180.)*np.cos( np.arctan2(yy, xx) )

wcs = WCS(hdu.header).slice([nx,ny])                         # calculate RA, Dec
bmaj=hdu.header['BMAJ']; bmin=hdu.header['BMIN']; bpa=hdu.header['BPA'] # Read beam parameter
bmaj=(bmaj*u.degree).to(u.arcsec).value; bmin=(bmin*u.degree).to(u.arcsec).value # Convert from degree to arcsec
cell=abs(hdu.header['CDELT2']); Cell=(cell*u.degree).to(u.arcsec).value # Read pixel size in arcsec

fig=plt.figure(num=None, figsize=(8,6), dpi=100, facecolor='w', edgecolor='k')
ax = fig.add_subplot(projection=wcs)
im=ax.imshow(1e-5*vrot.reshape((nx,ny)),origin='lower',cmap="jet",vmax=1.0,vmin=-1.0)
#im=ax.imshow(zz.reshape((nx,ny)),origin='lower',cmap="jet")
TT = ax.contour(zz.reshape((nx,ny)),[0.1,0.5,1.0,3.0],linestyles='solid',colors='gray')
plt.clabel(TT)
ax.set_xlabel('RA',fontsize=15)
ax.set_ylabel('Dec',fontsize=15)
#ax.text(0.95, 0.95, freq[i], ha='right', va='top', transform=ax.transAxes, color="w",fontsize=15)
ax.tick_params(axis='both',which='both',length=4,width=1,labelsize=12,direction='in',color='w')
#ax.margins(x=-0.375,y=-0.375)
cbar=plt.colorbar(im, shrink=0.9)
cbar.set_label(r'$\mathrm{(km\ s^{-1})^{2}}$', size=10)
plt.savefig('./Flared_emitting_layer_vkep_Rmatrix.pdf', bbox_inches='tight', pad_inches=0.1,dpi=100)
#plt.show()
plt.clf()
#'''
# =======================================================================================
# vector projection methods
# =======================================================================================
# Set sky plane xy
x=np.arange(-nx/2.,nx/2.)*abs(hdu.header['CDELT1'])*3600.*d_pc/100. # in au
y=np.arange(-ny/2.,ny/2.)*abs(hdu.header['CDELT2'])*3600.*d_pc/100. # in au
qq = np.meshgrid(x,y)
x_sky,y_sky = np.ravel(qq[0]), np.ravel(qq[1])

# Disk coordinate vector
x_disk, y_disk = 2.*x_sky, 2.*y_sky
r = np.hypot(x_disk,y_disk)
z_disk = ( z0 * np.power(r, psi) + z1 * np.power(r, phi) ) # z(r) = z0(r/1'')**psi - z1(r/1'')**phi

# Rotation velocity vector
vkep_disk = np.sqrt(GG*mstar/np.hypot(r,z_disk)/au/d_pc)
vkep_disk[np.isnan(vkep_disk)] = 0.0
theta = np.arctan2(y_disk,x_disk)
vx_disk = vkep_disk*np.sin(theta); vy_disk = vkep_disk*np.cos(theta); vz_disk = np.zeros_like(vx_disk)

# inclination: +z moves to +y axis
# DPA: +x moves to +y axis
ux_sky, uy_sky, uz_sky = rotated_axis(inc, DPA)
print(ux_sky, uy_sky, uz_sky)

# Calculate the projected (x', y', z') & (vx', vy', vz')
xx = vector_proj(x_disk, y_disk, z_disk, ux_sky)
yy = vector_proj(x_disk, y_disk, z_disk, uy_sky)
zz = vector_proj(x_disk, y_disk, z_disk, uz_sky)
vz = vector_proj(vx_disk, vy_disk, vz_disk, uz_sky)
#rr = np.hypot(xx,yy)
#l, xx, yy, zz, rr

z_prj = griddata((xx,yy),z_disk,(x_sky,y_sky),method='linear')
vrot = griddata((xx,yy),vz,(x_sky,y_sky),method='linear')

# Plotting
wcs = WCS(hdu.header).slice([nx,ny])                         # calculate RA, Dec
bmaj=hdu.header['BMAJ']; bmin=hdu.header['BMIN']; bpa=hdu.header['BPA'] # Read beam parameter
bmaj=(bmaj*u.degree).to(u.arcsec).value; bmin=(bmin*u.degree).to(u.arcsec).value # Convert from degree to arcsec
cell=abs(hdu.header['CDELT2']); Cell=(cell*u.degree).to(u.arcsec).value # Read pixel size in arcsec

fig=plt.figure(num=None, figsize=(8,6), dpi=100, facecolor='w', edgecolor='k')
ax = fig.add_subplot(projection=wcs)
im=ax.imshow(1e-5*vrot.reshape((nx,ny)),origin='lower',cmap="jet",vmax=1.0,vmin=-1.0)
#im=ax.imshow(z_prj.reshape((nx,ny)),origin='lower',cmap="jet")
TT = ax.contour(z_prj.reshape((nx,ny)),[0.1,0.5,1.0,3.0],linestyles='solid',colors='gray')
plt.clabel(TT)
ax.set_xlabel('RA',fontsize=15)
ax.set_ylabel('Dec',fontsize=15)
#ax.text(0.95, 0.95, freq[i], ha='right', va='top', transform=ax.transAxes, color="w",fontsize=15)
ax.tick_params(axis='both',which='both',length=4,width=1,labelsize=12,direction='in',color='w')
#ax.margins(x=-0.375,y=-0.375)
cbar=plt.colorbar(im, shrink=0.9)
cbar.set_label(r'$\mathrm{(km\ s^{-1})^{2}}$', size=10)
plt.savefig('./Flared_emitting_layer_vkep_vector.pdf', bbox_inches='tight', pad_inches=0.1,dpi=100)
#plt.show()
plt.clf()


# =======================================================================================
# Compare with GoFish
# =======================================================================================

# Moment maps derived from the Azimuthally averaged spectra
cube = imagecube(fdir+fitsname)
rvals, tvals, _ = cube.disk_coords(x0=dxc,y0=dyc,inc=inc,PA=270.-DPA,z0=z0,psi=psi,z1=z1,phi=phi)
r_au = rvals * d_pc
vmod = cube._keplerian(rpnts=rvals,mstar=0.63,dist=160.0, inc=inc,z0=z0, psi=psi, z1=z1, phi=phi)#, r_taper=260./160.
vmod *= np.cos(tvals)*1e-3; vmod[np.isnan(vmod)] = 0.0
#bmaj = cube.header['BMAJ']*3.6e3; bmin = cube.header['BMIN']*3.6e3; bpa = cube.header['BPA']
#pixsize_x = abs(cube.header['CDELT1']*3.6e3); pixsize_y = abs(cube.header['CDELT2']*3.6e3)
#gaussian_2D_kernel = Gaussian2DKernel(bmaj/pixsize_x/np.sqrt(8*np.log(2)),bmin/pixsize_y/np.sqrt(8*np.log(2)),bpa/180.*np.pi,x_size=151, y_size=151)
# Convolve the image through the 2D beam
#vmod = convolve(vmod,gaussian_2D_kernel,boundary='fill',fill_value='0',normalize_kernel=True)

fig=plt.figure(num=None, figsize=(8,6), dpi=100, facecolor='w', edgecolor='k')
ax = fig.add_subplot(projection=wcs)
im=ax.imshow(vmod,origin='lower',vmax=1.0,vmin=-1.0,cmap="jet")
#im=ax.imshow(_,origin='lower',cmap="jet")
TT = ax.contour(_,[0.1,0.5,1.0,3.0],linestyles='solid',colors='gray')
plt.clabel(TT)
ax.set_xlabel('RA',fontsize=15)
ax.set_ylabel('Dec',fontsize=15)
#ax.text(0.95, 0.95, freq[i], ha='right', va='top', transform=ax.transAxes, color="w",fontsize=15)
ax.tick_params(axis='both',which='both',length=4,width=1,labelsize=12,direction='in',color='w')
#ax.margins(x=-0.375,y=-0.375)
cbar=plt.colorbar(im, shrink=0.9)
cbar.set_label(r'$\mathrm{(km\ s^{-1})^{2}}$', size=10)
plt.savefig('./Flared_emitting_layer_vkep_gofish.pdf', bbox_inches='tight', pad_inches=0.1,dpi=100)
#plt.show()
plt.clf()
"""
# Plot teardrop figure
r_asc, velax, spectra, scatter  = cube.radial_spectra(inc=inc,PA=DPA,mstar=0.64,dist=160.0,dr=0.05,z0=z0,psi=psi,z1=z1,phi=phi)
plt.figure(figsize=(10,6))
plt.pcolormesh(1e-3*velax,r_asc*d_pc,spectra,norm=colors.LogNorm(vmin=1e-2*spectra.max(),vmax=spectra.max()),cmap='gist_heat',shading='gouraud',rasterized=True)
#plt.xlim(-1,2.4)
#plt.ylim(-2,-0.25)
plt.xlabel(r'V$_{radio}$ (km/s)', fontsize=15)
plt.ylabel('R (arcsec)',fontsize=15)
#plt.legend(prop={'size':15},loc=1)
plt.tick_params(which='both',length=6,width=1.5)
cbar = plt.colorbar() #ticks=[1e4,1e5,1e6,1e7])
cbar.set_label(r'$Jy\ beam^{-1}$', size=10)
plt.savefig('./'+mole+'_teardrop.pdf', bbox_inches='tight', pad_inches=0.1,dpi=100)
"""
"""
fig=plt.figure(num=None, figsize=(8,6), dpi=100, facecolor='w', edgecolor='k')
ax = fig.add_subplot(projection=wcs)
im=ax.imshow(abs(1e-5*vrot.reshape((nx,ny))/vmod.reshape((nx,ny))),origin='lower',vmax=1.5,vmin=0.5,cmap="jet")
ax.set_xlabel('RA',fontsize=15)
ax.set_ylabel('Dec',fontsize=15)
#ax.text(0.95, 0.95, freq[i], ha='right', va='top', transform=ax.transAxes, color="w",fontsize=15)
ax.tick_params(axis='both',which='both',length=4,width=1,labelsize=12,direction='in',color='w')
#ax.margins(x=-0.375,y=-0.375)
cbar=plt.colorbar(im, shrink=0.9)
cbar.set_label(r'$\mathrm{(km\ s^{-1})^{2}}$', size=10)
plt.savefig('./Flared_emitting_layer_vkep_ratio.pdf', bbox_inches='tight', pad_inches=0.1,dpi=100)
#plt.show()
plt.clf()
"""


#"""
# =======================================================================================
# vector projection methods for disk winds
# =======================================================================================
# Read grid file -----------------------------------------------------------------------------
fnameread      = 'amr_grid_1500au.inp'
nr, ntheta, nphi, grids = read_grid_inp(fnameread)
ri = grids[0:nr+1]; thetai = grids[nr+1:nr+ntheta+2]; phii = grids[nr+ntheta+2:]
rc       = 0.5 * ( ri[0:nr] + ri[1:nr+1] )
thetac   = 0.5 * ( thetai[0:ntheta] + thetai[1:ntheta+1] )
phic     = 0.5 * ( phii[0:nphi] + phii[1:nphi+1] )
qq       = np.meshgrid(rc,thetac,indexing='ij')
rr       = qq[0]    # cm
tt       = qq[1]     # rad
zzr       = np.pi/2.e0 - qq[1]     # rad

# Read veolcity field files   ------------------------------------
vr, vtheta, vphi = read_vel_input('gas_velocity_1500au.inp')
vr=vr.reshape((ntheta,nr)).T*1e-5
vtheta=vtheta.reshape((ntheta,nr)).T*1e-5
vphi=vphi.reshape((ntheta,nr)).T*1e-5
vp = np.ravel(np.hypot(vr,vtheta))  # vwind*cos(45) in km/s unit
rs = np.ravel(rr*np.sin(tt)/au); zs = np.ravel(rr*np.cos(tt)/au) # in au unit

# Set sky plane xy
x=np.arange(-nx/2.,nx/2.)*abs(hdu.header['CDELT1'])*3600.*d_pc # in au
y=np.arange(-ny/2.,ny/2.)*abs(hdu.header['CDELT2'])*3600.*d_pc # in au
qq = np.meshgrid(x,y)
x_sky,y_sky = np.ravel(qq[0]), np.ravel(qq[1])
x_disk, y_disk = 2.*x_sky, 2.*y_sky
r_disk = np.hypot(x_disk,y_disk)
#z_disk = ( z0 * np.power(r_disk/d_pc, psi) + z1 * np.power(r_disk/d_pc, phi) )*np.exp(-np.power(r_disk / 750., 1.0)) *r_disk # z(r) = z0(r/1'')**psi - z1(r/1'')**phi
z_disk = r_disk*0.35*(1.+(r_disk/290.)**1.4)*np.exp(-np.power(r_disk / 300., 2.0)) #zcn = rs * 0.35* (1+ (rs/290./au )**1.4 )

# Rotation velocity vector
vkep_disk = np.sqrt(GG*mstar/np.hypot(r_disk,z_disk)/au)
vkep_disk[np.isnan(vkep_disk)] = 0.0
theta = np.arctan2(y_disk,x_disk)
vkx_disk = vkep_disk*np.sin(theta); vky_disk = vkep_disk*np.cos(theta); vkz_disk = np.zeros_like(vkx_disk)

# Wind Velocity vector
vz_disk = griddata((rs,zs), vp, (r_disk, z_disk) ) # vz
vz_disk[np.where(np.isnan(vz_disk))] = 0.0
vx_disk = vz_disk*np.cos(np.arctan2(y_sky,x_sky))
vy_disk = vz_disk*np.sin(np.arctan2(y_sky,x_sky))

# inclination: +z moves to +y axis
# DPA: +x moves to +y axis
ux_sky, uy_sky, uz_sky = rotated_axis(inc, DPA)
print(ux_sky, uy_sky, uz_sky)
# Calculate the projected (x', y', z') & (vx', vy', vz')
xx = vector_proj(x_disk, y_disk, z_disk, ux_sky)
yy = vector_proj(x_disk, y_disk, z_disk, uy_sky)
zz = vector_proj(x_disk, y_disk, z_disk, uz_sky)
vz = vector_proj(vx_disk, vy_disk, vz_disk, uz_sky)
vkep = vector_proj(vkx_disk, vky_disk, vkz_disk, uz_sky)
#rr = np.hypot(xx,yy)
#l, xx, yy, zz, rr

z_prj = griddata((xx,yy),z_disk,(x_sky,y_sky),method='linear')
vw = griddata((xx,yy),vz,(x_sky,y_sky),method='linear'); vw[np.isnan(vw)] = 0.0
vk = griddata((xx,yy),vkep,(x_sky,y_sky),method='linear'); vk[np.isnan(vk)] = 0.0

# Convolution of V field by a beam size
gaussian_2D_kernel = Gaussian2DKernel(bmaj/pixsize_x/np.sqrt(8*np.log(2)),bmin/pixsize_y/np.sqrt(8*np.log(2)),bpa/180.*np.pi,x_size=151, y_size=151)
# Convolve the image through the 2D beam
vw_conv = convolve(vw.reshape((nx,ny)),gaussian_2D_kernel,boundary='extend',normalize_kernel=True)

# Plotting
wcs = WCS(hdu.header).slice([nx,ny])                         # calculate RA, Dec
bmaj=hdu.header['BMAJ']; bmin=hdu.header['BMIN']; bpa=hdu.header['BPA'] # Read beam parameter
bmaj=(bmaj*u.degree).to(u.arcsec).value; bmin=(bmin*u.degree).to(u.arcsec).value # Convert from degree to arcsec
cell=abs(hdu.header['CDELT2']); Cell=(cell*u.degree).to(u.arcsec).value # Read pixel size in arcsec

disk = Ellipse((250,250), 240./150.*100, 240./150.*100*np.cos(inc*np.pi/180.), angle=31., edgecolor='white', facecolor='none',ls='--')  # Keplerian disk
ring = Ellipse((250,250), 520./150.*100, 520./150.*100*np.cos(inc*np.pi/180.), angle=31., edgecolor='cyan', facecolor='none',ls='--')   # Outer envelope boundary
fig=plt.figure(num=None, figsize=(8,6), dpi=100, facecolor='w', edgecolor='k')
ax = fig.add_subplot(projection=wcs)
im=ax.imshow(vw_conv,origin='lower',cmap="jet",vmax=0.5,vmin=0.0)
#im=ax.imshow(vwind.reshape((nx,ny)),origin='lower',cmap="jet")
#TT = ax.contour(z_prj.reshape((nx,ny)),[0.1,0.2,0.3,0.5,1.0],linestyles='solid',colors='gray')
#plt.clabel(TT)
#TT2 = ax.contour(rr.reshape((nx,ny)),[10,30,100,300],linestyles='--',colors='w')
#plt.clabel(TT2)
ax.set_xlabel('RA',fontsize=15)
ax.set_ylabel('Dec',fontsize=15)
ax.add_patch(disk)
ax.add_patch(ring)
#ax.text(0.95, 0.95, freq[i], ha='right', va='top', transform=ax.transAxes, color="w",fontsize=15)
ax.tick_params(axis='both',which='both',length=4,width=1,labelsize=12,direction='in',color='w')
ax.margins(x=-0.2,y=-0.2)
cbar=plt.colorbar(im, shrink=0.9)
cbar.set_label(r'$\mathrm{(km\ s^{-1})^{2}}$', size=10)
plt.savefig('./Flared_emitting_layer_vwind_vector.pdf', bbox_inches='tight', pad_inches=0.1,dpi=100)
#plt.show()
plt.clf()

disk = Ellipse((250,250), 240./150.*100, 240./150.*100*np.cos(inc*np.pi/180.), angle=31., edgecolor='white', facecolor='none',ls='--')  # Keplerian disk
ring = Ellipse((250,250), 520./150.*100, 520./150.*100*np.cos(inc*np.pi/180.), angle=31., edgecolor='cyan', facecolor='none',ls='--')   # Outer envelope boundary
fig=plt.figure(num=None, figsize=(8,6), dpi=100, facecolor='w', edgecolor='k')
ax = fig.add_subplot(projection=wcs)
im=ax.imshow(1e-5*vk.reshape((nx,ny)),origin='lower',cmap="jet",vmax=1.0,vmin=-1.0)
#im=ax.imshow(vwind.reshape((nx,ny)),origin='lower',cmap="jet")
#TT = ax.contour(z_prj.reshape((nx,ny)),[0.1,0.2,0.3,0.5,1.0],linestyles='solid',colors='gray')
#plt.clabel(TT)
#TT2 = ax.contour(rr.reshape((nx,ny)),[10,30,100,300],linestyles='--',colors='w')
#plt.clabel(TT2)
ax.set_xlabel('RA',fontsize=15)
ax.set_ylabel('Dec',fontsize=15)
ax.add_patch(disk)
ax.add_patch(ring)
#ax.text(0.95, 0.95, freq[i], ha='right', va='top', transform=ax.transAxes, color="w",fontsize=15)
ax.tick_params(axis='both',which='both',length=4,width=1,labelsize=12,direction='in',color='w')
ax.margins(x=-0.25,y=-0.25)
cbar=plt.colorbar(im, shrink=0.9)
cbar.set_label(r'$\mathrm{(km\ s^{-1})^{2}}$', size=10)
plt.savefig('./Flared_emitting_layer_vrot_vector.pdf', bbox_inches='tight', pad_inches=0.1,dpi=100)
#plt.show()
plt.clf()

disk = Ellipse((250,250), 240./150.*100, 240./150.*100*np.cos(inc*np.pi/180.), angle=31., edgecolor='white', facecolor='none',ls='--')  # Keplerian disk
ring = Ellipse((250,250), 520./150.*100, 520./150.*100*np.cos(inc*np.pi/180.), angle=31., edgecolor='cyan', facecolor='none',ls='--')   # Outer envelope boundary
fig=plt.figure(num=None, figsize=(8,6), dpi=100, facecolor='w', edgecolor='k')
ax = fig.add_subplot(projection=wcs)
im=ax.imshow(abs(vw/(1e-5*vk)).reshape((nx,ny)),origin='lower',cmap="jet",vmax=0.3,vmin=0.0)
#im=ax.imshow(vwind.reshape((nx,ny)),origin='lower',cmap="jet")
#TT = ax.contour(z_prj.reshape((nx,ny)),[0.1,0.2,0.3,0.5,1.0],linestyles='solid',colors='gray')
#plt.clabel(TT)
#TT2 = ax.contour(rr.reshape((nx,ny)),[10,30,100,300],linestyles='--',colors='w')
#plt.clabel(TT2)
ax.set_xlabel('RA',fontsize=15)
ax.set_ylabel('Dec',fontsize=15)
ax.add_patch(disk)
ax.add_patch(ring)
#ax.text(0.95, 0.95, freq[i], ha='right', va='top', transform=ax.transAxes, color="w",fontsize=15)
ax.tick_params(axis='both',which='both',length=4,width=1,labelsize=12,direction='in',color='w')
ax.margins(x=-0.25,y=-0.25)
cbar=plt.colorbar(im, shrink=0.9)
cbar.set_label(r'$\mathrm{(km\ s^{-1})^{2}}$', size=10)
plt.savefig('./Flared_emitting_layer_wind_rot_ratio_vector.pdf', bbox_inches='tight', pad_inches=0.1,dpi=100)
#plt.show()
plt.clf()

#"""
