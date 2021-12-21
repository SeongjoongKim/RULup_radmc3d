from matplotlib.colors import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.gridspec as gridspec
from astropy.io import fits
from astropy import units as u
from scipy.interpolate import interp1d
#import bettermoments.collapse_cube as bm
#from Model_setup_subroutines import *
import argparse
from gofish import imagecube
from scipy.optimize import curve_fit
from astropy.convolution import Gaussian2DKernel, convolve
import os

# =======================================================================================
# Functions
# =======================================================================================
def Gaussian(x,a0,b0,c0,d0):
    return a0*np.exp(-(x-b0)**2/(2*c0**2)) +d0 #+ a1*np.exp(-(x-b0-d0)**2/(2*c0**2))


def Double(x,a0,b0,c0,a1,d0):
    return a0*np.exp(-(x-b0)**2/(2*c0**2)) + a1*np.exp(-(x-b0+0.6784)**2/(2*c0**2)) + d0 #+ a1*np.exp(-(x-b0-d0)**2/(2*c0**2))


def Spec_fit(data, vax, rr, tt, vmodel, r_min, r_max, PA_min, PA_max):
    r_mask = np.logical_and(rr >= r_min, rr <= r_max)
    PA_mask = np.logical_and(abs(tt) >= PA_min*np.pi/180., abs(tt) <= PA_max*np.pi/180.)
    v_mask = np.logical_and(vmodel >= -5.0e3, vmodel <= 5.0e3)
    mask = r_mask*PA_mask*v_mask
    nv, ny, nx = data.shape
    ndpnts = np.sum(mask)
    a = np.zeros(ndpnts) ; b = np.zeros(ndpnts); c = np.zeros(ndpnts); d = np.zeros(ndpnts)
    idx = np.zeros(ndpnts, dtype = np.intc); idy = np.zeros(ndpnts, dtype = np.intc)
    k = 0
    for i in range(nx):
        for j in range(ny):
            if mask[j,i]:
                idx[k] = i ; idy[k] = j
                try:
                    [ a[k], b[k], c[k], d[k] ], pcov1 = curve_fit(Gaussian, vax, data[:,j,i], p0=[0.01,vmodel[j,i]*1e-3,0.2,0], bounds=([0,vmodel[j,i]*1e-3-1.0,0,0 ] ,[0.5,vmodel[j,i]*1e-3+1.0,2.0,1.0] ), absolute_sigma=True)
                except RuntimeError:
                    print("Error - curve_fit failed at pixel [{:03d},{:03d}]".format(i,j))
                k += 1
                
    return a,b,c,d,idx,idy


def Spec_double(data, vax, rr, tt, vmodel, r_min, r_max, PA_min, PA_max):
    r_mask = np.logical_and(rr >= r_min, rr <= r_max)
    PA_mask = np.logical_and(abs(tt) >= PA_min*np.pi/180., abs(tt) <= PA_max*np.pi/180.)
    v_mask = np.logical_and(vmodel >= -5.0e3, vmodel <= 5.0e3)
    mask = r_mask*PA_mask*v_mask
    nv, ny, nx = data.shape
    ndpnts = np.sum(mask)
    a = np.zeros(ndpnts) ; b = np.zeros(ndpnts); c = np.zeros(ndpnts); d = np.zeros(ndpnts)
    aa = np.zeros(ndpnts) #; bb = np.zeros(ndpnts); cc = np.zeros(ndpnts)
    idx = np.zeros(ndpnts, dtype = np.intc); idy = np.zeros(ndpnts, dtype = np.intc)
    k = 0
    for i in range(nx):
        for j in range(ny):
            if mask[j,i]:
                idx[k] = i ; idy[k] = j #bb[k], cc[k],
                try:
                    [ a[k], b[k], c[k], aa[k],  d[k] ], pcov1 = curve_fit(Double, vax, data[:,j,i], p0=[0.01,vmodel[j,i]*1e-3,0.2,0.01,0], bounds=([1e-3,-5.0,0,1e-3,0] ,[1.0,5.0,2.0,0.5,1.0] ), absolute_sigma=True)
                except RuntimeError:
                    print("Error - curve_fit failed at pixel [{:03d},{:03d}]".format(i,j))
                k += 1
            
    return a,b,c,aa,d,idx,idy  #bb,cc,


# =======================================================================================
# Parameter setup
# =======================================================================================
parser = argparse.ArgumentParser()
parser.add_argument('-mole', default='None', type=str, help='The molecular line name. Default is None.')
parser.add_argument('-bmaj', default='bmaj5', type=str, help='The beam size. Default is bmaj5.')
parser.add_argument('-double', default='F', type=str, help='The fitting of line shape. "F" is a single Gaussian and "T" is double peak shape. Default is "F".')
#parser.add_argument('-rmin', default=0.0, type=float, help='The inner radius in arcsec unit. Default is 0.0')
#parser.add_argument('-rmax', default=1.0, type=float, help='The outer radius in arcsec unit. Default is 1.0')
#parser.add_argument('-PAmin', default=-180.0, type=float, help='The lower PA in degree unit. Default is -180.0')
#parser.add_argument('-PAmax', default=180.0, type=float, help='The upper PA in degree unit. Default is 180.0')
args = parser.parse_args()

mole = args.mole  #'C18O_2-1'
bmaj = args.bmaj  #'bmaj5'
double = args.double
inc = 25.0
r_min = 0.75;  r_max = 0.85
DPA = 121.0; PA_min = 45; PA_max = 135
dxc = 0.00; dyc = 0.00
z0 = 0.0;psi = 1.0; z1 = 0.0; phi = 1.0

dirName = '../spec_figure/CN0.35_Cw1e-4/'+mole+'_r{:4.2f}-{:4.2f}_PA{:03d}-{:03d}_'.format(r_min,r_max,PA_min,PA_max)+bmaj+'/'
if double == 'T':
    dirName = '../spec_figure/CN0.35_Cw1e-4/'+mole+'_r{:4.2f}-{:4.2f}_PA{:03d}-{:03d}_'.format(r_min,r_max,PA_min,PA_max)+bmaj+'_double/'

if not os.path.exists(dirName):
    os.mkdir(dirName)
    print("Directory " , dirName ,  " Created ")
else:
    print("Directory " , dirName ,  " already exists")

# =======================================================================================
# No wind model
# =======================================================================================
fdir = '/Users/kimsj/Documents/RADMC-3D/radmc3d-2.0/RU_Lup_test/Automatics/Fin_script/fiducial/'
fitsname = 'RULup_'+mole+'_fiducial_'+bmaj+'.fits'  #
cube = imagecube(fdir+fitsname)
#for ii in range(cube.header['NAXIS3']): cube.data[ii,:,:] = cube.data[ii,:,:].T
rvals, tvals, _ = cube.disk_coords(x0=dxc,y0=dyc,inc=inc,PA=DPA,z0=z0,psi=psi,z1=z1,phi=phi)
vmod = cube._keplerian(rpnts=rvals,mstar=0.63,dist=160.0, inc=inc)
vmod *= np.cos(tvals)
b_maj = cube.header['BMAJ']*3.6e3; b_min = cube.header['BMIN']*3.6e3; bpa = cube.header['BPA']
pixsize_x = abs(cube.header['CDELT1']*3.6e3); pixsize_y = abs(cube.header['CDELT2']*3.6e3)
gaussian_2D_kernel = Gaussian2DKernel(b_maj/pixsize_x/np.sqrt(8*np.log(2)),b_min/pixsize_y/np.sqrt(8*np.log(2)),bpa/180.*np.pi,x_size=151, y_size=151)
# Convolve the image through the 2D beam
vmod = convolve(vmod,gaussian_2D_kernel,boundary='fill',fill_value='0',normalize_kernel=True)
if double == 'F':
    a,b,c,d,idx,idy = Spec_fit(cube.data,cube.velax,rvals,tvals,vmod,r_min,r_max,PA_min,PA_max)
    with open(dirName+'Spec_fit_rmin{:4.2f}_rmax{:4.2f}_params_nowind.dat'.format(r_min,r_max),'w+') as f:
        f.write('Averages of a b c d\n')
        f.write('%13.6e %13.6e %13.6e %13.6e\n'%(a.mean(),b.mean(),c.mean(),d.mean()))
        f.write('a b c d idx idy\n')                   # Include r,theta, phi in coordinates
        for i in range(len(a)):
            f.write('%13.6e %13.6e %13.6e %13.6e %4d %4d\n'%(a[i],b[i],c[i],d[i],idx[i],idy[i]))
if double == 'T':
    a,b,c,a2,d,idx,idy = Spec_double(cube.data,cube.velax,rvals,tvals,vmod,r_min,r_max,PA_min,PA_max) #b2,c2,
    with open(dirName+'Spec_fit_rmin{:4.2f}_rmax{:4.2f}_params_nowind_double.dat'.format(r_min,r_max),'w+') as f:
        f.write('Averages of a0 b0 c0 a1 d0\n') #b1 c1
        f.write('%13.6e %13.6e %13.6e %13.6e %13.6e \n'%(a.mean(),b.mean(),c.mean(),a2.mean(),d.mean())) #%13.6e %13.6e # ,b2.mean(),c2.mean()
        f.write('a0 b0 c0 a1 d0 idx idy\n')                   # Include r,theta, phi in coordinates  b1 c1
        for i in range(len(a)):
            f.write('%13.6e %13.6e %13.6e %13.6e %13.6e %4d %4d\n'%(a[i],b[i],c[i],a2[i],d[i],idx[i],idy[i])) # %13.6e %13.6e #b2[i],c2[i],

#"""
# =======================================================================================
# Plotting the masking regions and spectra
# =======================================================================================
#shifted_cube = cube.shifted_cube(r_min=r_min,r_max=r_max,inc=inc,PA=DPA, mstar=0.63, dist=160.0, x0=dxc, y0=dyc)
x1,y1,dy1 = cube.average_spectrum(r_min=r_min, r_max=r_max, dr=0.1, inc=inc,PA=DPA, mstar=0.63, dist=160.0, x0=dxc, y0=dyc ,PA_min = PA_min, PA_max = PA_max,mask_frame='disk', abs_PA=True)  # Minor axis 1
fig, ax = plt.subplots()
ax.imshow(cube.data[100,:,:], origin='lower',extent=cube.extent, vmin=cube.data.min(), vmax=cube.data[100,:,:].max()*0.95)
cube.plot_mask(ax=ax, r_min=r_min, r_max=r_max, PA_min=PA_min, PA_max=PA_max, inc=inc, PA=DPA, mask_frame='disk',abs_PA=True,x0=dxc,y0=dyc)
ax.set_ylabel('$\Delta DEC$ [arcsec]')
ax.set_xlabel('$\Delta RA$ [arcsec]')
ax.margins(x=-0.4,y=-0.4)
plt.savefig(dirName+'RULup_'+mole+'_GoFish_map.pdf', bbox_inches='tight', pad_inches=0.1)
#plt.show()
plt.close()

fig, ax2 = plt.subplots()
#ax2.set_xlim(-4,4)
ax2.set_xlabel('Vel [km/s]')
ax2.set_ylabel('I$_{model}$ [Jy/beam]')
ax2.plot(x1,y1,'k',label='+90')
#ax2.plot(x,y2,'k--',label='-90')
#ax2.legend(prop={'size':12},loc=0)
plt.savefig(dirName+'RULup_'+mole+'_GoFish_spec.pdf', bbox_inches='tight', pad_inches=0.1)
#plt.show()
plt.close()
#"""

# =======================================================================================
# Wind model
# =======================================================================================
fdir2 = '/Users/kimsj/Documents/RADMC-3D/radmc3d-2.0/RU_Lup_test/Automatics/Fin_script/fiducial_wind/'
windname = 'RULup_'+mole+'_fiducial_wind_CN0.35_Cw1e-4_'+bmaj+'.fits'#
cube_wind = imagecube(fdir2+windname)
#for ii in range(cube_wind.header['NAXIS3']): cube_wind.data[ii,:,:] = cube_wind.data[ii,:,:].T
#cube_wind.velax *= cube.header['CDELT3']/cube_wind.header['CDELT3']
rvalsw, tvalsw, _ = cube_wind.disk_coords(x0=dxc,y0=dyc,inc=inc,PA=DPA,z0=z0,psi=psi,z1=z1,phi=phi)
vmodw = cube_wind._keplerian(rpnts=rvalsw,mstar=0.63,dist=160.0, inc=inc)
vmodw *= np.cos(tvalsw)
b_maj = cube_wind.header['BMAJ']*3.6e3; b_min = cube_wind.header['BMIN']*3.6e3; bpa = cube_wind.header['BPA']
pixsize_x = abs(cube_wind.header['CDELT1']*3.6e3); pixsize_y = abs(cube_wind.header['CDELT2']*3.6e3)
gaussian_2D_kernel = Gaussian2DKernel(b_maj/pixsize_x/np.sqrt(8*np.log(2)),b_min/pixsize_y/np.sqrt(8*np.log(2)),bpa/180.*np.pi,x_size=151, y_size=151)
# Convolve the image through the 2D beam
vmodw = convolve(vmodw,gaussian_2D_kernel,boundary='fill',fill_value='0',normalize_kernel=True)
if double == 'F':
    aw,bw,cw,dw,idxw,idyw = Spec_fit(cube_wind.data,cube_wind.velax,rvalsw,tvalsw,vmodw,r_min,r_max,PA_min,PA_max)
    with open(dirName+'Spec_fit_rmin{:4.2f}_rmax{:4.2f}_params_wind.dat'.format(r_min,r_max),'w+') as f:
        f.write('Averages of a b c d\n')
        f.write('%13.6e %13.6e %13.6e %13.6e\n'%(aw.mean(),bw.mean(),cw.mean(),dw.mean()))
        f.write('a b c d idx idy\n')                   # Include r,theta, phi in coordinates
        for i in range(len(a)):
            f.write('%13.6e %13.6e %13.6e %13.6e %4d %4d\n'%(a[i],b[i],c[i],d[i],idx[i],idy[i]))
if double == 'T':
    aw,bw,cw,aw2,dw,idxw,idyw = Spec_double(cube_wind.data,cube_wind.velax,rvalsw,tvalsw,vmodw,r_min,r_max,PA_min,PA_max) #bw2,cw2,
    with open(dirName+'Spec_fit_rmin{:4.2f}_rmax{:4.2f}_params_wind_double.dat'.format(r_min,r_max),'w+') as f:
        f.write('Averages of a0 b0 c0 a1 d0\n') #b1 c1
        f.write('%13.6e %13.6e %13.6e %13.6e %13.6e \n'%(aw.mean(),bw.mean(),cw.mean(),aw2.mean(),dw.mean())) #%13.6e %13.6e # ,bw2.mean(),cw2.mean()
        f.write('a0 b0 c0 a1 d0 idx idy\n')                   # Include r,theta, phi in coordinates   b1 c1
        for i in range(len(a)):
            f.write('%13.6e %13.6e %13.6e %13.6e %13.6e %4d %4d\n'%(aw[i],bw[i],cw[i],aw2[i],dw[i],idxw[i],idyw[i])) # %13.6e %13.6e # bw2[i],cw2[i],

# =======================================================================================
# Observation
# =======================================================================================
#obsname = '/Users/kimsj/Documents/RU_Lup/Fin_fits/C18O_all_selfcal_p1st_wc_matched_cube500.fits'
obsname = '/Users/kimsj/Documents/RU_Lup/Fin_fits/'+mole+'_selfcal_wc_matched_cube500.fits'
cube_obs = imagecube(obsname)
nv_obs = cube_obs.header['NAXIS3']
#for ii in range(cube_obs.header['NAXIS3']): cube_obs.data[ii,:,:] = cube_obs.data[ii,:,:].T
vel_obs = cube_obs.velax*1e-3 -4.5 #0.52 +np.arange(nv_obs)*0.084 - 5.0
#x3,y3,dy3 = cube_obs.average_spectrum(r_min=r_min, r_max=r_max, dr=0.1, inc=inc,PA=DPA, mstar=0.63, dist=160.0, x0=dxc, y0=dyc ,PA_min = PA_min, PA_max = PA_max,mask_frame='disk', abs_PA=True)  # Minor axis 1


# =======================================================================================
# Plotting the spectrum at each pixels included in the mask
# =======================================================================================
#del_x = [-2,-1,0,1,2]
#del_y = [-2,-1,0,1,2]
#del_x = [5,6,7,8,9]
#del_y = [-3,-4,-5,-6]
#del_x = [-5,-6,-7,-8,-9]
#del_y = [3,4,5,6]
#del_x = np.arange(-10,11,1)
#del_y = np.arange(-10,11,1)
#xc=250; yc=250
del_x = idx#[3]
del_y = idy#[-5]
for i in range(50): #len(del_x)
    ix = del_x[i];  iy=del_y[i]
    #ix = xc+del_x[i];  iy=yc+del_y[i]
    #print( (iy-yc+1)/(ix-xc+1)*180./np.pi )
    spec = cube.data[:,iy,ix]
    #shifted_spec = interp1d(x1-vmod[iy,ix]*1e-3,spec,bounds_error=False)(x1)
    spec_wind = cube_wind.data[:,iy,ix]
    #shifted_wind = interp1d(x1-vmod[iy,ix]*1e-3,spec_wind,bounds_error=False)(x1)
    spec_obs = cube_obs.data[:,iy,ix]
    #print(rr_au[iy,ix],v_los[iy,ix])
    fig = plt.figure(figsize=(12,5))
    grdspec = gridspec.GridSpec(ncols=2, nrows=1, figure=fig)
    #plt.title('Radius ~ '+str(rr_au[iy,ix])+', V$_{cen}$ ~ '+str(v_los[iy,ix]) , fontsize=15 )
    ax1 = fig.add_subplot(grdspec[0])
    #plt.ylim(spec.min()*0.9,spec.max()*1.1)
    ax1.set_xlim(-6,6)
    ax1.set_xlabel('Vel [km/s]')
    ax1.set_ylabel('I$_{model}$ [Jy/beam]')
    ax1.plot(x1,spec,'r',label='Wind X')
    #ax1.plot(x1,shifted_spec,'r--')
    if double == 'F': ax1.plot(x1,Gaussian(x1,a[i],b[i],c[i],d[i]),'c-.')
    if double == 'T': ax1.plot(x1,Double(x1,a[i],b[i],c[i],a2[i],d[i]),'c-.') #b2[i],c2[i],
    ax1.plot(x1,spec_wind,'b',label='Wind O')
    #ax1.plot(x1,shifted_wind,'b--')
    if double == 'F': ax1.plot(x1,Gaussian(x1,aw[i],bw[i],cw[i],dw[i]),'m-.')
    if double == 'T': ax1.plot(x1,Double(x1,aw[i],bw[i],cw[i],aw2[i],dw[i]),'m-.') #,bw2[i],cw2[i]
    ax1.axvline(x=vmod[iy,ix]*1e-3,lw=0.8,ls='--',color='g')
    ax1.axvline(x=0.0,lw=0.8,ls='--',color='k')
    ax3 = ax1.twinx()
    ax3.set_xlim(-6,6)
    ax3.set_ylabel('I$_{obs}$ [Jy/beam]')
    ax3.plot(vel_obs,spec_obs,'k--',label='Obs',lw=0.3)
    #ax1.plot(vel,spec/spec.max(),'k',label='Wind X')
    #ax1.plot(vel,spec_wind/spec_wind.max(),'r',label='Wind O')
    #ax1.plot(vel_obs,spec_obs/spec_obs.max(),'k--')
    ax1.legend(prop={'size':12},loc=0)
    ax1.text(0.95, 1.05, r'R = {:4.2f} au, $\theta$ = {:4.2f}, V$_C$ ~ {:4.2f} km/s'.format(rvals[iy,ix]*160.0,tvals[iy,ix]*180./np.pi, vmod[iy,ix]*1e-3), ha='right', va='top', transform=ax1.transAxes, color="k",fontsize=13)
    ax2 = fig.add_subplot(grdspec[1])
    im=ax2.imshow(vmod*1e-3,origin='lower',vmax=3,vmin=-3,cmap='jet')
    ax2.scatter(ix,iy,c='m')
    #ax2.text(0.75, 1.05, 'V$_{cen}$ ~ '+str(v_los[iy,ix]), ha='right', va='top', transform=ax2.transAxes, color="k",fontsize=13)
    ax2.margins(x=-0.4,y=-0.4)
    cbar=plt.colorbar(im, shrink=0.9)
    plt.savefig(dirName+'RULup_'+mole+'_['+str(ix)+','+str(iy)+']_model_spec.pdf', bbox_inches='tight', pad_inches=0.1)
    #plt.show()
    plt.close()





# ====================================================================================================================
"""
for i in range(len(del_x)):
    for j in range(len(del_y)):
        ix = xc+del_x[i];  iy=yc+del_y[j]
        #print( (iy-yc+1)/(ix-xc+1)*180./np.pi )
        spec = cube.data[:,iy,ix]
        shifted_spec = interp1d(x1-vmod[iy,ix]*1e-3,spec,bounds_error=False)(x1)
        spec_wind = cube_wind.data[:,iy,ix]
        shifted_wind = interp1d(x1-vmod[iy,ix]*1e-3,spec_wind,bounds_error=False)(x1)
        spec_obs = cube_obs.data[:,iy,ix]
        #print(rr_au[ix,iy],v_los[ix,iy])
        fig = plt.figure(figsize=(12,5))
        grdspec = gridspec.GridSpec(ncols=2, nrows=1, figure=fig)
        #plt.title('Radius ~ '+str(rr_au[ix,iy])+', V$_{cen}$ ~ '+str(v_los[ix,iy]) , fontsize=15 )
        ax1 = fig.add_subplot(grdspec[0])
        #plt.ylim(spec.min()*0.9,spec.max()*1.1)
        ax1.set_xlim(-4,4)
        ax1.set_xlabel('Vel [km/s]')
        ax1.set_ylabel('I$_{model}$ [Jy/beam]')
        ax1.plot(x1,spec,'r',label='Wind X')
        ax1.plot(x1,shifted_spec,'r--')
        ax1.plot(x1,Gaussian(x1,a[i],b[i],c[i],d[i]),'c-.')
        ax1.plot(x1,spec_wind,'b',label='Wind O')
        ax1.plot(x1,shifted_wind,'b--')
        ax1.axvline(x=vmod[iy,ix]*1e-3,lw=0.8,ls='--',color='g')
        ax1.axvline(x=0.0,lw=0.8,ls='--',color='k')
        ax3 = ax1.twinx()
        ax3.set_xlim(-4,4)
        ax3.set_ylabel('I$_{obs}$ [Jy/beam]')
        ax3.plot(vel_obs,spec_obs,'k--',label='Obs',lw=0.3)
        #ax1.plot(vel,spec/spec.max(),'k',label='Wind X')
        #ax1.plot(vel,spec_wind/spec_wind.max(),'r',label='Wind O')
        #ax1.plot(vel_obs,spec_obs/spec_obs.max(),'k--')
        ax1.legend(prop={'size':12},loc=0)
        ax1.text(0.95, 1.05, r'R = {:4.2f} au, $\theta$ = {:4.2f}, V$_C$ ~ {:4.2f} km/s'.format(rvals[iy,ix]*160.0,tvals[iy,ix]*180./np.pi, vmod[iy,ix]*1e-3), ha='right', va='top', transform=ax1.transAxes, color="k",fontsize=13)
        ax2 = fig.add_subplot(grdspec[1])
        im=ax2.imshow(vmod*1e-3,origin='lower',vmax=3,vmin=-3,cmap='jet')
        ax2.scatter(ix,iy,c='m')
        #ax2.text(0.75, 1.05, 'V$_{cen}$ ~ '+str(v_los[ix,iy]), ha='right', va='top', transform=ax2.transAxes, color="k",fontsize=13)
        ax2.margins(x=-0.45,y=-0.45)
        cbar=plt.colorbar(im, shrink=0.9)
        plt.savefig('/Users/seongjoongkim/Documents/RADMC-3D/radmc3d-2.0/RU_Lup_test/Automatics/spec_figure/B1/RULup_'+mole+'_['+str(ix)+','+str(iy)+']_model_spec.pdf', bbox_inches='tight', pad_inches=0.1)
        #plt.show()
        plt.close()
"""
"""
# ==================================    Manual calculations of V field    ========================================================
# Constants
c = 2.99792458e5 # [km/s]
au  = 1.49598e13     # Astronomical Unit       [cm]
pc  = 3.08572e18     # Parsec                  [cm]
GG  = 6.67408e-08    # Gravitational constant  [cm^3/g/s^2]
ms  = 1.98892e33     # Solar mass              [g]
d_pc = 160.0
inc = 25.0; dpa = 180.0+31.0

# Read fitsfile and set parameters
hdu = fits.open(fdir+fitsname)[0]
hdr = hdu.header
data = hdu.data
bmaj = hdr['BMAJ']*3600.;  bmin = hdr['BMIN']*3600.;  bpa = hdr['BPA']
nx = hdr['NAXIS1'];  ny = hdr['NAXIS2'];  nv = hdr['NAXIS3']
pxsize_x = abs(hdr['CDELT1']*3600);  pxsize_y = abs(hdr['CDELT2']*3600)
vel = -3.0+np.arange(nv)*0.06  #np.arange(nv)*hdr['CDELT3'] + hdr['CRVAL3']
xc = int(hdr['CRPIX1']-1);  yc = int(hdr['CRPIX2']-1)
#for i in range(nv): data[i,:,:] = data[i,:,:].T           #  The data structure of [vel, DEC, RA] No transpose to swap RA and Dec
data = np.swapaxes(data, 0, 2)

# Read wind included model fits
fdir2 = '/Users/seongjoongkim/Documents/RADMC-3D/radmc3d-2.0/RU_Lup_test/Automatics/fiducial_all_wind/S_test/'
test2 = 'wind_DSHARP_H0.1_D0.1_S0.001_3e6'#_B1_CN0.25
windname = 'RULup_'+mole+'_3-2_'+test2+'_RADMC3D_model_cube_bmaj0.05.fits'
hdu3 = fits.open(fdir2+windname)[0]
hdr3 = hdu3.header
wind = hdu3.data
nv_wind = hdr3['NAXIS3']
#for i in range(nv_wind): wind[i,:,:] = wind[i,:,:].T       #  The data structure of [vel, DEC, RA] No transpose to swap RA and Dec
wind = np.swapaxes(wind, 0, 2)

# Read observational data
#obsname = '/Users/seongjoongkim/Documents/RU_Lup/Fin_fits/C18O_all_selfcal_p1st_wc_matched_cube500.fits'
obsname = '/Users/seongjoongkim/Documents/RU_Lup/Fin_fits/CN_3-2_selfcal_wc_matched_cube500.fits'
hdu2 = fits.open(obsname)[0]
hdr2 = hdu2.header
obs = hdu2.data
nv_obs = hdr2['NAXIS3']
for i in range(nv_obs): obs[0,i,:,:] = obs[0,i,:,:].T       #  Transpose for matching the data structure of [Stokes, vel, DEC, RA]
obs = np.swapaxes(obs, 0, 3); obs = np.swapaxes(obs, 1, 2)  #  Switching the axes to [RA, DEC, vel, Stokes]
vel_obs = 0.52 +np.arange(nv_obs)*0.084 - 5.0

# Rotated grid setting
x = np.arange(nx)-xc    ; y = np.arange(ny)-yc
qq = np.meshgrid(x, y)
xx = qq[0]; yy = qq[1]
xx_rot = xx*np.cos(dpa*np.pi/180.) + yy*np.sin(dpa*np.pi/180.)   # Pixel
yy_rot = ( -xx*np.sin(dpa*np.pi/180.) + yy*np.cos(dpa*np.pi/180.) ) / np.cos(inc*np.pi/180.0)  # Pixel
rr_au = np.sqrt( xx_rot**2 +yy_rot**2)*pxsize_x*d_pc  # au
# Velocity field
mstar    = 0.63*ms
v_los = np.sqrt(GG*mstar/rr_au/au) *np.sin(inc*np.pi/180.) *(xx_rot/np.sqrt(xx_rot**2+yy_rot**2)) *1e-5  # sqrt(GM/r) sin(i) cos(theta) Rosenfeld et al 2013, ApJ, 774, 16
#v_los = v_los.T
#print(xx_rot[xc-2:xc+3,yc-2:yc+3], yy_rot[xc-2:xc+3,yc-2:yc+3])
#print(rr_au[xc-2:xc+3,yc-2:yc+3])
#print(v_los[xc-2:xc+3,yc-2:yc+3])


"""
"""
zoom_data = cube.data[:,230:270,230:270]
zoom_rvals = rvals[230:270,230:270]
zoom_tvals = tvals[230:270,230:270]
zoom_rau = zoom_rvals*160.0
zoom_angle = zoom_tvals*180.0/np.pi
zoom_vmod = vmod[230:270,230:270]*1e-3
idx, idy = np.where(zoom_rvals<0.2)

shifted = np.empty(100)
shifted = interp1d(x-zoom_vmod[10,10],zoom_data[:,10,10],bounds_error=False)(x)

"""
