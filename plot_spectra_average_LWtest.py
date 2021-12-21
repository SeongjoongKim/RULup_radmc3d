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


def Double(x,a0,b0,c0,a1,b1,d0):
    return a0*np.exp(-(x-b0)**2/(2*c0**2)) + a1*np.exp(-(x-b0+b1)**2/(2*c0**2)) + d0 #+ a1*np.exp(-(x-b0-d0)**2/(2*c0**2))


def Spec_fit(data, vax, rr, tt, vmodel, r_min, r_max, PA_min, PA_max):
    r_mask = np.logical_and(rr >= r_min, rr <= r_max)
    PA_mask = np.logical_and(abs(tt) >= PA_min*np.pi/180., abs(tt) <= PA_max*np.pi/180.)
    v_mask = np.logical_and(vmodel >= -5.0e3, vmodel <= 5.0e3)
    mask = r_mask*PA_mask*v_mask
    nv, ny, nx = data.shape
    ndpnts = np.sum(mask)
    shifted = np.zeros(nv)
    ndata = np.ones(nv)*ndpnts
    for i in range(nx):
        for j in range(ny):
            if mask[j,i]:
                shifted_spec = interp1d(vax - 1e-3*vmodel[j, i], data[:, j, i], bounds_error=False)(vax)
                ndata[np.isnan(shifted_spec)] -= 1
                shifted_spec[np.isnan(shifted_spec)] = 0.0
                shifted += shifted_spec
    shifted /= ndata
    try:
        [ a, b, c, d ], pcov1 = curve_fit(Gaussian, vax, shifted, p0=[0.01,0,0.2,0], absolute_sigma=True)
    except RuntimeError:
        print("Error - curve_fit failed")
    #print(ndpnts)
    return shifted, a,b,c,d


def Spec_double(data, vax, rr, tt, vmodel, r_min, r_max, PA_min, PA_max):
    r_mask = np.logical_and(rr >= r_min, rr <= r_max)
    PA_mask = np.logical_and(abs(tt) >= PA_min*np.pi/180., abs(tt) <= PA_max*np.pi/180.)
    v_mask = np.logical_and(vmodel >= -5.0e3, vmodel <= 5.0e3)
    mask = r_mask*PA_mask*v_mask
    nv, ny, nx = data.shape
    ndpnts = np.sum(mask)
    shifted = np.zeros(nv)
    ndata = np.ones(nv)*ndpnts
    for i in range(nx):
        for j in range(ny):
            if mask[j,i]:
                shifted_spec = interp1d(vax - 1e-3*vmodel[i, j], data[:, i, j], bounds_error=False)(vax)
                ndata[np.isnan(shifted_spec)] -= 1
                shifted_spec[np.isnan(shifted_spec)] = 0.0
                shifted += shifted_spec
    shifted /= ndata
    try:
        [ a, b, c, aa, bb, d ], pcov1 = curve_fit(Double, vax, data[:,i,j], p0=[0.01,0,0.2,0.01,0.7,0], absolute_sigma=True)
    except RuntimeError:
        print("Error - curve_fit failed")
    #print(ndpnts)
    return shifted, a,b,c,aa,bb,d  #bb,cc,


# =======================================================================================
# Parameter setup
# =======================================================================================
parser = argparse.ArgumentParser()
parser.add_argument('-mole', default='None', type=str, help='The molecular line name. Default is None.')
parser.add_argument('-bmaj', default='bmaj5', type=str, help='The beam size. Default is bmaj5.')
parser.add_argument('-double', default='F', type=str, help='The fitting of line shape. "F" is a single Gaussian and "T" is double peak shape. Default is "F".')
parser.add_argument('-rmin', default=0.0, type=float, help='The inner radius in arcsec unit. Default is 0.0')
parser.add_argument('-rmax', default=1.0, type=float, help='The outer radius in arcsec unit. Default is 1.0')
parser.add_argument('-PAmin', default=0, type=int, help='The lower PA in degree unit. Default is -180')
parser.add_argument('-PAmax', default=180, type=int, help='The upper PA in degree unit. Default is 180')
args = parser.parse_args()

mole = args.mole  #'C18O_2-1'
bmaj = args.bmaj  #'bmaj5'
double = args.double
inc = 25.0
r_min = args.rmin; r_max = args.rmax
PA_min = args.PAmin; PA_max = args.PAmax
#r_min = 0.15;  r_max = 0.25
DPA = 121.0#; PA_min = 45; PA_max = 135   # PA from -180 to 180
dxc = 0.00; dyc = 0.00
z0 = 0.00;psi = 1.0; z1 = 0.0; phi = 1.0

dirName = '../spec_figure/Line_width_test/'+mole+'_r{:4.2f}-{:4.2f}_PA{:03d}-{:03d}_azi-aver_'.format(r_min,r_max,PA_min,PA_max)+bmaj+'_LWtest/'
if double == 'T':
    dirName = '../spec_figure/Line_width_test/'+mole+'_r{:4.2f}-{:4.2f}_PA{:03d}-{:03d}_azi-aver_'.format(r_min,r_max,PA_min,PA_max)+bmaj+'_LWtest_double/'

if not os.path.exists(dirName):
    os.mkdir(dirName)
    print("Directory " , dirName ,  " Created ")
else:
    print("Directory " , dirName ,  " already exists")

# =======================================================================================
# Thermal only
# =======================================================================================
fdir = '/Users/kimsj/Documents/RADMC-3D/radmc3d-2.0/RU_Lup_test/Automatics/Fin_script/fiducial_wind/'
fitsname = 'RULup_'+mole+'_fiducial_wind_Thermal_'+bmaj+'.fits'  #
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
    I_ave,a,b,c,d = Spec_fit(cube.data,cube.velax,rvals,tvals,vmod,r_min,r_max,PA_min,PA_max)
    with open(dirName+'Spec_fit_rmin{:4.2f}_rmax{:4.2f}_params_Thermal.dat'.format(r_min,r_max),'w+') as f:
        f.write('Gaussian parameters of a b c d\n')
        f.write('%13.6e %13.6e %13.6e %13.6e\n'%(a,b,c,d))
        f.write('Keplerian corrected spectrum\n')                   # Include r,theta, phi in coordinates
        for i in range(len(I_ave)):
            f.write('%13.6e \n'%(I_ave[i]))
if double == 'T':
    I_ave,a,b,c,a2,b2,d = Spec_double(cube.data,cube.velax,rvals,tvals,vmod,r_min,r_max,PA_min,PA_max) #b2,c2,
    with open(dirName+'Spec_fit_rmin{:4.2f}_rmax{:4.2f}_params_Thermal_double.dat'.format(r_min,r_max),'w+') as f:
        f.write('Gaussian parameters of a0 b0 c0 a1 d0\n') #b1 c1
        f.write('%13.6e %13.6e %13.6e %13.6e %13.6e %13.6e \n'%(a,b,c,a2,b2,d)) #%13.6e %13.6e # ,b2.mean(),c2.mean()
        f.write('Keplerian corrected spectrum\n')                   # Include r,theta, phi in coordinates  b1 c1
        for i in range(len(I_ave)):
            f.write('%13.6e \n'%(I_ave[i])) # %13.6e %13.6e #b2[i],c2[i],


x1,y1,dy1 = cube.average_spectrum(r_min=r_min, r_max=r_max, dr=0.1, inc=inc,PA=DPA, mstar=0.63, dist=160.0, x0=dxc, y0=dyc ,PA_min = PA_min, PA_max = PA_max,mask_frame='disk', abs_PA=True)  # Minor axis 1

# =======================================================================================
# Thermal + Turb
# =======================================================================================
fdir = '/Users/kimsj/Documents/RADMC-3D/radmc3d-2.0/RU_Lup_test/Automatics/Fin_script/fiducial_wind/'
fitsname = 'RULup_'+mole+'_fiducial_wind_Thermal+Turb_'+bmaj+'.fits'  #
cube_turb = imagecube(fdir+fitsname)
#for ii in range(cube.header['NAXIS3']): cube.data[ii,:,:] = cube.data[ii,:,:].T
rvalst, tvalst, _ = cube_turb.disk_coords(x0=dxc,y0=dyc,inc=inc,PA=DPA,z0=z0,psi=psi,z1=z1,phi=phi)
vmodt = cube_turb._keplerian(rpnts=rvalst,mstar=0.63,dist=160.0, inc=inc)
vmodt *= np.cos(tvalst)
b_maj = cube_turb.header['BMAJ']*3.6e3; b_min = cube_turb.header['BMIN']*3.6e3; bpa = cube_turb.header['BPA']
pixsize_x = abs(cube_turb.header['CDELT1']*3.6e3); pixsize_y = abs(cube_turb.header['CDELT2']*3.6e3)
gaussian_2D_kernel = Gaussian2DKernel(b_maj/pixsize_x/np.sqrt(8*np.log(2)),b_min/pixsize_y/np.sqrt(8*np.log(2)),bpa/180.*np.pi,x_size=151, y_size=151)
# Convolve the image through the 2D beam
vmodt = convolve(vmodt,gaussian_2D_kernel,boundary='fill',fill_value='0',normalize_kernel=True)
if double == 'F':
    I_avet,at,bt,ct,dt = Spec_fit(cube_turb.data,cube_turb.velax,rvalst,tvalst,vmodt,r_min,r_max,PA_min,PA_max)
    with open(dirName+'Spec_fit_rmin{:4.2f}_rmax{:4.2f}_params_Thermal+Turb.dat'.format(r_min,r_max),'w+') as f:
        f.write('Gaussian parameters of a b c d\n')
        f.write('%13.6e %13.6e %13.6e %13.6e\n'%(at,bt,ct,dt))
        f.write('Keplerian corrected spectrum\n')                   # Include r,theta, phi in coordinates
        for i in range(len(I_avet)):
            f.write('%13.6e \n'%(I_avet[i]))
if double == 'T':
    I_avet,at,bt,ct,a2t,b2t,dt = Spec_double(cube_turb.data,cube_turb.velax,rvalst,tvalst,vmodt,r_min,r_max,PA_min,PA_max) #b2,c2,
    with open(dirName+'Spec_fit_rmin{:4.2f}_rmax{:4.2f}_params_Thermal+Turb_double.dat'.format(r_min,r_max),'w+') as f:
        f.write('Gaussian parameters of a0 b0 c0 a1 d0\n') #b1 c1
        f.write('%13.6e %13.6e %13.6e %13.6e %13.6e %13.6e \n'%(at,bt,ct,a2t,b2t,dt)) #%13.6e %13.6e # ,b2.mean(),c2.mean()
        f.write('Keplerian corrected spectrum\n')                   # Include r,theta, phi in coordinates  b1 c1
        for i in range(len(I_avet)):
            f.write('%13.6e \n'%(I_avet[i])) # %13.6e %13.6e #b2[i],c2[i],


x2,y2,dy2 = cube_turb.average_spectrum(r_min=r_min, r_max=r_max, dr=0.1, inc=inc,PA=DPA, mstar=0.63, dist=160.0, x0=dxc, y0=dyc ,PA_min = PA_min, PA_max = PA_max,mask_frame='disk', abs_PA=True)  # Minor axis 1

# =======================================================================================
# Thermal + Wind
# =======================================================================================
fdir = '/Users/kimsj/Documents/RADMC-3D/radmc3d-2.0/RU_Lup_test/Automatics/Fin_script/fiducial_wind/'
fitsname = 'RULup_'+mole+'_fiducial_wind_Thermal+wind_'+bmaj+'.fits'  #
cube_wind = imagecube(fdir+fitsname)
#for ii in range(cube.header['NAXIS3']): cube.data[ii,:,:] = cube.data[ii,:,:].T
rvalsw, tvalsw, _ = cube_wind.disk_coords(x0=dxc,y0=dyc,inc=inc,PA=DPA,z0=z0,psi=psi,z1=z1,phi=phi)
vmodw = cube_wind._keplerian(rpnts=rvalsw,mstar=0.63,dist=160.0, inc=inc)
vmodw *= np.cos(tvalsw)
b_maj = cube_wind.header['BMAJ']*3.6e3; b_min = cube_wind.header['BMIN']*3.6e3; bpa = cube_wind.header['BPA']
pixsize_x = abs(cube_wind.header['CDELT1']*3.6e3); pixsize_y = abs(cube_wind.header['CDELT2']*3.6e3)
gaussian_2D_kernel = Gaussian2DKernel(b_maj/pixsize_x/np.sqrt(8*np.log(2)),b_min/pixsize_y/np.sqrt(8*np.log(2)),bpa/180.*np.pi,x_size=151, y_size=151)
# Convolve the image through the 2D beam
vmodw = convolve(vmodw,gaussian_2D_kernel,boundary='fill',fill_value='0',normalize_kernel=True)
if double == 'F':
    I_avew,aw,bw,cw,dw = Spec_fit(cube_wind.data,cube_wind.velax,rvalsw,tvalsw,vmodw,r_min,r_max,PA_min,PA_max)
    with open(dirName+'Spec_fit_rmin{:4.2f}_rmax{:4.2f}_params_Thermal+wind.dat'.format(r_min,r_max),'w+') as f:
        f.write('Gaussian parameters of a b c d\n')
        f.write('%13.6e %13.6e %13.6e %13.6e\n'%(aw,bw,cw,dw))
        f.write('Keplerian corrected spectrum\n')                   # Include r,theta, phi in coordinates
        for i in range(len(I_avew)):
            f.write('%13.6e \n'%(I_avew[i]))
if double == 'T':
    I_avew,aw,bw,cw,a2w,b2w,dw = Spec_double(cube_wind.data,cube_wind.velax,rvalsw,tvalsw,vmodw,r_min,r_max,PA_min,PA_max) #b2,c2,
    with open(dirName+'Spec_fit_rmin{:4.2f}_rmax{:4.2f}_params_Thermal+wind_double.dat'.format(r_min,r_max),'w+') as f:
        f.write('Gaussian parameters of a0 b0 c0 a1 d0\n') #b1 c1
        f.write('%13.6e %13.6e %13.6e %13.6e %13.6e %13.6e \n'%(aw,bw,cw,a2w,b2w,dw)) #%13.6e %13.6e # ,b2.mean(),c2.mean()
        f.write('Keplerian corrected spectrum\n')                   # Include r,theta, phi in coordinates  b1 c1
        for i in range(len(I_avew)):
            f.write('%13.6e \n'%(I_avew[i])) # %13.6e %13.6e #b2[i],c2[i],


x3,y3,dy3 = cube_wind.average_spectrum(r_min=r_min, r_max=r_max, dr=0.1, inc=inc,PA=DPA, mstar=0.63, dist=160.0, x0=dxc, y0=dyc ,PA_min = PA_min, PA_max = PA_max,mask_frame='disk', abs_PA=True)  # Minor axis 1

# =======================================================================================
# Fiducial wind
# =======================================================================================
fdir = '/Users/kimsj/Documents/RADMC-3D/radmc3d-2.0/RU_Lup_test/Automatics/Fin_script/fiducial_wind/'
fitsname = 'RULup_'+mole+'_fiducial_wind_'+bmaj+'.fits'  #
cube_fidu = imagecube(fdir+fitsname)
#for ii in range(cube.header['NAXIS3']): cube.data[ii,:,:] = cube.data[ii,:,:].T
rvalsf, tvalsf, _ = cube_fidu.disk_coords(x0=dxc,y0=dyc,inc=inc,PA=DPA,z0=z0,psi=psi,z1=z1,phi=phi)
vmodf = cube_fidu._keplerian(rpnts=rvalsf,mstar=0.63,dist=160.0, inc=inc)
vmodf *= np.cos(tvalsf)
b_maj = cube_fidu.header['BMAJ']*3.6e3; b_min = cube_fidu.header['BMIN']*3.6e3; bpa = cube_fidu.header['BPA']
pixsize_x = abs(cube_fidu.header['CDELT1']*3.6e3); pixsize_y = abs(cube_fidu.header['CDELT2']*3.6e3)
gaussian_2D_kernel = Gaussian2DKernel(b_maj/pixsize_x/np.sqrt(8*np.log(2)),b_min/pixsize_y/np.sqrt(8*np.log(2)),bpa/180.*np.pi,x_size=151, y_size=151)
# Convolve the image through the 2D beam
vmodf = convolve(vmodf,gaussian_2D_kernel,boundary='fill',fill_value='0',normalize_kernel=True)
if double == 'F':
    I_avef,af,bf,cf,df = Spec_fit(cube_fidu.data,cube_fidu.velax,rvalsf,tvalsf,vmodf,r_min,r_max,PA_min,PA_max)
    with open(dirName+'Spec_fit_rmin{:4.2f}_rmax{:4.2f}_params_fiducial.dat'.format(r_min,r_max),'w+') as f:
        f.write('Gaussian parameters of a b c d\n')
        f.write('%13.6e %13.6e %13.6e %13.6e\n'%(af,bf,cf,df))
        f.write('Keplerian corrected spectrum\n')                   # Include r,theta, phi in coordinates
        for i in range(len(I_avef)):
            f.write('%13.6e \n'%(I_avef[i]))
if double == 'T':
    I_avef,af,bf,cf,a2f,b2f,df = Spec_double(cube_fidu.data,cube_fidu.velax,rvalsf,tvalsf,vmodf,r_min,r_max,PA_min,PA_max) #b2,c2,
    with open(dirName+'Spec_fit_rmin{:4.2f}_rmax{:4.2f}_params_fiducial_double.dat'.format(r_min,r_max),'w+') as f:
        f.write('Gaussian parameters of a0 b0 c0 a1 d0\n') #b1 c1
        f.write('%13.6e %13.6e %13.6e %13.6e %13.6e %13.6e \n'%(af,bf,cf,a2f,b2f,df)) #%13.6e %13.6e # ,b2.mean(),c2.mean()
        f.write('Keplerian corrected spectrum\n')                   # Include r,theta, phi in coordinates  b1 c1
        for i in range(len(I_avef)):
            f.write('%13.6e \n'%(I_avef[i])) # %13.6e %13.6e #b2[i],c2[i],


x4,y4,dy4 = cube_fidu.average_spectrum(r_min=r_min, r_max=r_max, dr=0.1, inc=inc,PA=DPA, mstar=0.63, dist=160.0, x0=dxc, y0=dyc ,PA_min = PA_min, PA_max = PA_max,mask_frame='disk', abs_PA=True)  # Minor axis 1

# =======================================================================================
# Plotting the spectrum at each pixels included in the mask
# =======================================================================================
fig = plt.figure(figsize=(7,5))
grdspec = gridspec.GridSpec(ncols=1, nrows=1, figure=fig)
#plt.title('Radius ~ '+str(rr_au[iy,ix])+', V$_{cen}$ ~ '+str(v_los[iy,ix]) , fontsize=15 )
ax1 = fig.add_subplot(grdspec[0])
#plt.ylim(spec.min()*0.9,spec.max()*1.1)
ax1.set_xlim(-6,6)
ax1.set_xlabel('Vel [km/s]')
ax1.set_ylabel('I$_{model}$ [Jy/beam]')
ax1.plot(cube.velax,I_ave,'r',label='Ther')
ax1.plot(x1,y1,'r--')
if double == 'F': ax1.plot(x1,Gaussian(x1,a,b,c,d),'r-.',lw=0.5)
if double == 'T': ax1.plot(x1,Double(x1,a,b,c,a2,b2,d),'r-.',lw=0.5) #b2,c2,
ax1.plot(cube_turb.velax,I_avet,'g',label='Ther+Turb')
ax1.plot(x2,y2,'g--')
if double == 'F': ax1.plot(x1,Gaussian(x1,at,bt,ct,dt),'g-.',lw=0.5)
if double == 'T': ax1.plot(x1,Double(x1,at,bt,ct,a2t,b2t,dt),'g-.',lw=0.5) #,bw2,cw2
ax1.plot(cube_wind.velax,I_avew,'b',label='Ther+wind')
ax1.plot(x3,y3,'b--')
if double == 'F': ax1.plot(x1,Gaussian(x1,aw,bw,cw,dw),'b-.',lw=0.5)
if double == 'T': ax1.plot(x1,Double(x1,aw,bw,cw,a2w,b2w,dw),'b-.',lw=0.5) #,bw2,cw2
ax1.plot(cube_fidu.velax,I_avef,'k',label='fiducial')
ax1.plot(x4,y4,'k--')
if double == 'F': ax1.plot(x1,Gaussian(x1,aw,bw,cw,dw),'b-.',lw=0.5)
if double == 'T': ax1.plot(x1,Double(x1,aw,bw,cw,a2w,b2w,dw),'b-.',lw=0.5) #,bw2,cw2
#ax1.axvline(x=vmod[iy,ix]*1e-3,lw=0.8,ls='--',color='orange')
#ax1.axvline(x=vmodw[iy,ix]*1e-3,lw=0.8,ls='--',color='g')
ax1.axvline(x=0.0,lw=0.8,ls='--',color='k')
#ax3 = ax1.twinx()
#ax3.set_xlim(-6,6)
#ax3.set_ylabel('I$_{obs}$ [Jy/beam]')
#ax3.plot(vel_obs ,I_aveo,'k',label='Obs',lw=0.5)
#ax3.plot(x3*1e-3 - 4.5,y3,'k--',lw=0.5)
ax1.legend(prop={'size':12},loc=0)
#ax1.text(0.95, 1.05, r'R = {:4.2f} au, $\theta$ = {:4.2f}, V$_C$ ~ {:4.2f} km/s'.format(rvals[iy,ix]*160.0,tvals[iy,ix]*180./np.pi, vmod[iy,ix]*1e-3), ha='right', va='top', transform=ax1.transAxes, color="k",fontsize=13)
plt.savefig(dirName+'RULup_'+mole+'_rmin{:4.2f}_rmax{:4.2f}_model_spec.pdf'.format(r_min,r_max), bbox_inches='tight', pad_inches=0.1)
#plt.show()
plt.close()


