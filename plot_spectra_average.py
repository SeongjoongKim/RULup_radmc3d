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
#def Double(x,a0,b0,c0,a1,d0):
#    return a0*np.exp(-(x-b0)**2/(2*c0**2)) + a1*np.exp(-(x-b0+0.6784)**2/(2*c0**2)) + d0 #+ a1*np.exp(-(x-b0-d0)**2/(2*c0**2))


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
        [ aa, bb, cc, dd ], pcov1 = curve_fit(Gaussian, vax, shifted, p0=[0.01,0,0.2,0], absolute_sigma=True)
    except RuntimeError:
        print("Error - curve_fit failed")
        aa,bb,cc,dd = [0.01,0,0.2,0]
    #print(ndpnts)
    return shifted, aa,bb,cc,dd


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
        [ aa, bb, cc, aa2, bb2, dd ], pcov1 = curve_fit(Double, vax, data[:,i,j], p0=[0.01,0,0.2,0.005,0.7,0], absolute_sigma=True)
    except RuntimeError:
        print("Error - curve_fit failed")
        aa,bb,cc,aa2,bb2,dd = [0.01,0,0.2,0.005,0.7,0]
    #print(ndpnts)
    return shifted, aa,bb,cc,aa2,bb2,dd  #bb,cc,


# =======================================================================================
# Parameter setup
# =======================================================================================
parser = argparse.ArgumentParser()
parser.add_argument('-mole', default='None', type=str, help='The molecular line name. Default is None.')
parser.add_argument('-bmaj', default='bmaj5', type=str, help='The beam size. Default is bmaj5.')
parser.add_argument('-tname', default='None', type=str, help='Test name. Default is None.')
parser.add_argument('-double', default='F', type=str, help='The fitting of line shape. "F" is a single Gaussian and "T" is double peak shape. Default is "F".')
#parser.add_argument('-rmin', default=0.0, type=float, help='The inner radius in arcsec unit. Default is 0.0')
#parser.add_argument('-rmax', default=1.0, type=float, help='The outer radius in arcsec unit. Default is 1.0')
#parser.add_argument('-PAmin', default=-180.0, type=float, help='The lower PA in degree unit. Default is -180.0')
#parser.add_argument('-PAmax', default=180.0, type=float, help='The upper PA in degree unit. Default is 180.0')
args = parser.parse_args()

mole = args.mole  #'C18O_2-1'
bmaj = args.bmaj  #'bmaj5'
double = args.double
tname = args.tname
inc = 25.0
r0 = 0.05;  r1 = 0.85; dr =0.1
r_min = np.arange(r0,r1,dr); r_max = r_min + dr
DPA = 121.0; PA_min = 45; PA_max = 135   # PA from -180 to 180
dxc = 0.00; dyc = 0.00
z0 = 0.00;psi = 1.0; z1 = 0.0; phi = 1.0

n_gauss = 4
if double == 'T': n_gauss = 6
Summary_nowind = np.zeros((len(r_min), n_gauss))
Summary_wind = np.zeros((len(r_min),n_gauss))
Summary_obs = np.zeros((len(r_min),n_gauss))

# =======================================================================================
# File name setup
# =======================================================================================
# Set No wind model fits file
fdir = '/Users/kimsj/Documents/RADMC-3D/radmc3d-2.0/RU_Lup_test/Automatics/Fin_script/fiducial/'
fitsname = 'RULup_'+mole+'_'+tname+'_'+bmaj+'.fits'
if not os.path.exists(fdir+fitsname):
    fitsname = 'RULup_'+mole+'_fiducial_'+bmaj+'.fits'
cube = imagecube(fdir+fitsname)
#for ii in range(cube.header['NAXIS3']): cube.data[ii,:,:] = cube.data[ii,:,:].T

# Set Wind model fits file
fdir2 = '/Users/kimsj/Documents/RADMC-3D/radmc3d-2.0/RU_Lup_test/Automatics/Fin_script/fiducial_wind/'
windname = 'RULup_'+mole+'_'+tname+'_'+bmaj+'.fits'#
if not os.path.exists(fdir2+windname):
    windname = 'RULup_'+mole+'_fiducial_wind_'+bmaj+'.fits'#
cube_wind = imagecube(fdir2+windname)
#for ii in range(cube_wind.header['NAXIS3']): cube_wind.data[ii,:,:] = cube_wind.data[ii,:,:].T
#cube_wind.velax *= cube.header['CDELT3']/cube_wind.header['CDELT3']

# Set Observation fits file
#obsname = '/Users/kimsj/Documents/RU_Lup/Fin_fits/C18O_all_selfcal_p1st_wc_matched_cube500.fits'
obsname = '/Users/kimsj/Documents/RU_Lup/Fin_fits/'+mole+'_selfcal_wc_matched_cube500.fits'
cube_obs = imagecube(obsname)
nv_obs = cube_obs.header['NAXIS3']
vel_obs = cube_obs.velax*1e-3 - 5.0

# Output directory of Gaussian fitting parameter summary
outdir = '../spec_figure/'+tname+'/'
if not os.path.exists(outdir):
    os.mkdir(outdir)
    print("Directory " , outdir ,  " Created ")
else:
    print("Directory " , outdir ,  " already exists")

# =======================================================================================
# Making fittings of spectra and plotting
# =======================================================================================
for k in range(len(r_min)):
    dirName = outdir+mole+'_r{:4.2f}-{:4.2f}_PA{:03d}-{:03d}_azi-aver_'.format(r_min[k],r_max[k],PA_min,PA_max)+bmaj+'/'
    if double == 'T':
        dirName = outdir+mole+'_r{:4.2f}-{:4.2f}_PA{:03d}-{:03d}_azi-aver_'.format(r_min[k],r_max[k],PA_min,PA_max)+bmaj+'_double/'
    if not os.path.exists(dirName):
        os.mkdir(dirName)
        print("Directory " , dirName ,  " Created ")
    else:
        print("Directory " , dirName ,  " already exists")
    # =======================================================================================
    # Plotting the masking regions and spectra
    # =======================================================================================
    #shifted_cube = cube.shifted_cube(r_min=r_min[k],r_max=r_max[k],inc=inc,PA=DPA, mstar=0.63, dist=160.0, x0=dxc, y0=dyc)
    x1,y1,dy1 = cube.average_spectrum(r_min=r_min[k], r_max=r_max[k], dr=0.1, inc=inc,PA=DPA, mstar=0.63, dist=160.0, x0=dxc, y0=dyc ,PA_min = PA_min, PA_max = PA_max,mask_frame='disk', abs_PA=True)  # Minor axis 1
    fig, ax = plt.subplots()
    ax.imshow(cube.data[100,:,:].T, origin='lower',extent=cube.extent, vmin=cube.data.min(), vmax=cube.data[50,:,:].max()*0.95)
    cube.plot_mask(ax=ax, r_min=r_min[k], r_max=r_max[k], PA_min=PA_min, PA_max=PA_max, inc=inc, PA=DPA, mask_frame='disk',abs_PA=True,x0=dxc,y0=dyc)
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
    
    # =======================================================================================
    # No wind model
    # =======================================================================================
    rvals, tvals, _ = cube.disk_coords(x0=dxc,y0=dyc,inc=inc,PA=DPA,z0=z0,psi=psi,z1=z1,phi=phi)
    vmod = cube._keplerian(rpnts=rvals,mstar=0.63,dist=160.0, inc=inc)
    vmod *= np.cos(tvals)
    b_maj = cube.header['BMAJ']*3.6e3; b_min = cube.header['BMIN']*3.6e3; bpa = cube.header['BPA']
    pixsize_x = abs(cube.header['CDELT1']*3.6e3); pixsize_y = abs(cube.header['CDELT2']*3.6e3)
    gaussian_2D_kernel = Gaussian2DKernel(b_maj/pixsize_x/np.sqrt(8*np.log(2)),b_min/pixsize_y/np.sqrt(8*np.log(2)),bpa/180.*np.pi,x_size=151, y_size=151)
    # Convolve the image through the 2D beam
    vmod = convolve(vmod,gaussian_2D_kernel,boundary='fill',fill_value='0',normalize_kernel=True)
    if double == 'F':
        I_ave,a,b,c,d = Spec_fit(cube.data,cube.velax,rvals,tvals,vmod,r_min[k],r_max[k],PA_min,PA_max)
        Summary_nowind[k,:] = [a, b, c, d]
        with open(dirName+'Spec_fit_rmin{:4.2f}_rmax{:4.2f}_params_nowind.dat'.format(r_min[k],r_max[k]),'w+') as f:
            f.write('Gaussian parameters of a b c d\n')
            f.write('%13.6e %13.6e %13.6e %13.6e\n'%(a,b,c,d))
            f.write('Keplerian corrected spectrum\n')
            for i in range(len(I_ave)):
                f.write('%13.6e \n'%(I_ave[i]))
    if double == 'T':
        #I_ave,a,b,c,a2,d = Spec_double(cube.data,cube.velax,rvals,tvals,vmod,r_min[k],r_max[k],PA_min,PA_max)
        #Summary_nowind[k,:] = [a, b, c, a2, d]
        I_ave,a,b,c,a2,b2,d = Spec_double(cube.data,cube.velax,rvals,tvals,vmod,r_min[k],r_max[k],PA_min,PA_max)
        Summary_nowind[k,:] = [a, b, c, a2, b2, d]
        with open(dirName+'Spec_fit_rmin{:4.2f}_rmax{:4.2f}_params_nowind_double.dat'.format(r_min[k],r_max[k]),'w+') as f:
            f.write('Gaussian parameters of a0 b0 c0 a1 d0\n')
            #f.write('%13.6e %13.6e %13.6e %13.6e %13.6e \n'%(a,b,c,a2,d))
            f.write('%13.6e %13.6e %13.6e %13.6e %13.6e %13.6e\n'%(a,b,c,a2,b2,d))
            f.write('Keplerian corrected spectrum\n')
            for i in range(len(I_ave)):
                f.write('%13.6e \n'%(I_ave[i]))
    
    # =======================================================================================
    # Wind model
    # =======================================================================================
    rvalsw, tvalsw, _ = cube_wind.disk_coords(x0=dxc,y0=dyc,inc=inc,PA=DPA,z0=z0,psi=psi,z1=z1,phi=phi)
    vmodw = cube_wind._keplerian(rpnts=rvalsw,mstar=0.63,dist=160.0, inc=inc)
    vmodw *= np.cos(tvalsw)
    b_maj = cube_wind.header['BMAJ']*3.6e3; b_min = cube_wind.header['BMIN']*3.6e3; bpa = cube_wind.header['BPA']
    pixsize_x = abs(cube_wind.header['CDELT1']*3.6e3); pixsize_y = abs(cube_wind.header['CDELT2']*3.6e3)
    gaussian_2D_kernel = Gaussian2DKernel(b_maj/pixsize_x/np.sqrt(8*np.log(2)),b_min/pixsize_y/np.sqrt(8*np.log(2)),bpa/180.*np.pi,x_size=151, y_size=151)
    # Convolve the image through the 2D beam
    vmodw = convolve(vmodw,gaussian_2D_kernel,boundary='fill',fill_value='0',normalize_kernel=True)
    if double == 'F':
        I_avew,aw,bw,cw,dw = Spec_fit(cube_wind.data,cube_wind.velax,rvalsw,tvalsw,vmodw,r_min[k],r_max[k],PA_min,PA_max)
        Summary_wind[k,:] = [aw, bw, cw, dw]
        with open(dirName+'Spec_fit_rmin{:4.2f}_rmax{:4.2f}_params_wind.dat'.format(r_min[k],r_max[k]),'w+') as f:
            f.write('Gaussian parameters of a b c d\n')
            f.write('%13.6e %13.6e %13.6e %13.6e\n'%(aw,bw,cw,dw))
            f.write('Keplerian corrected spectrum\n')
            for i in range(len(I_avew)):
                f.write('%13.6e \n'%(I_avew[i]))
    if double == 'T':
        #I_avew,aw,bw,cw,aw2,dw = Spec_double(cube_wind.data,cube_wind.velax,rvalsw,tvalsw,vmodw,r_min[k],r_max[k],PA_min,PA_max)
        #Summary_wind[k,:] = [aw, bw, cw, aw2, dw]
        I_avew,aw,bw,cw,aw2,bw2,dw = Spec_double(cube_wind.data,cube_wind.velax,rvalsw,tvalsw,vmodw,r_min[k],r_max[k],PA_min,PA_max)
        Summary_wind[k,:] = [aw, bw, cw, aw2, bw2,dw]
        with open(dirName+'Spec_fit_rmin{:4.2f}_rmax{:4.2f}_params_wind_double.dat'.format(r_min[k],r_max[k]),'w+') as f:
            f.write('Gaussian parameters of a0 b0 c0 a1 d0\n')
            #f.write('%13.6e %13.6e %13.6e %13.6e %13.6e \n'%(aw,bw,cw,aw2,dw))
            f.write('%13.6e %13.6e %13.6e %13.6e %13.6e %13.6e\n'%(aw,bw,cw,aw2,bw2,dw))
            f.write('Keplerian corrected spectrum\n')
            for i in range(len(I_avew)):
                f.write('%13.6e \n'%(I_avew[i]))

    x2,y2,dy2 = cube_wind.average_spectrum(r_min=r_min[k], r_max=r_max[k], dr=0.1, inc=inc,PA=DPA, mstar=0.63, dist=160.0, x0=dxc, y0=dyc ,PA_min = PA_min, PA_max = PA_max,mask_frame='disk', abs_PA=True)  # Minor axis 1

    # =======================================================================================
    # Observation
    # =======================================================================================
    rvalso, tvalso, _ = cube_obs.disk_coords(x0=dxc,y0=dyc,inc=inc,PA=DPA,z0=z0,psi=psi,z1=z1,phi=phi)
    vmodo = cube_obs._keplerian(rpnts=rvalso,mstar=0.63,dist=160.0, inc=inc)
    vmodo *= np.cos(tvalso)
    b_maj = cube_obs.header['BMAJ']*3.6e3; b_min = cube_obs.header['BMIN']*3.6e3; bpa = cube_obs.header['BPA']
    pixsize_x = abs(cube_obs.header['CDELT1']*3.6e3); pixsize_y = abs(cube_obs.header['CDELT2']*3.6e3)
    gaussian_2D_kernel = Gaussian2DKernel(b_maj/pixsize_x/np.sqrt(8*np.log(2)),b_min/pixsize_y/np.sqrt(8*np.log(2)),bpa/180.*np.pi,x_size=151, y_size=151)
    # Convolve the image through the 2D beam
    vmodo = convolve(vmodo,gaussian_2D_kernel,boundary='fill',fill_value='0',normalize_kernel=True)
    if double == 'F':
        I_aveo,ao,bo,co,do = Spec_fit(cube_obs.data,vel_obs,rvalso,tvalso,vmodo,r_min[k],r_max[k],PA_min,PA_max)
        Summary_obs[k,:] = [ao, bo, co, do]
        with open(dirName+'Spec_fit_rmin{:4.2f}_rmax{:4.2f}_params_obs.dat'.format(r_min[k],r_max[k]),'w+') as f:
            f.write('Gaussian parameters of a b c d\n')
            f.write('%13.6e %13.6e %13.6e %13.6e\n'%(ao,bo,co,do))
            f.write('Keplerian corrected spectrum\n')
            for i in range(len(I_aveo)):
                f.write('%13.6e \n'%(I_aveo[i]))
    if double == 'T':
        #I_aveo,ao,bo,co,ao2,do = Spec_double(cube_obs.data,vel_obs,rvalso,tvalso,vmodo,r_min[k],r_max[k],PA_min,PA_max)
        #Summary_obs[k,:] = [ao, bo, co, ao2, do]
        I_aveo,ao,bo,co,ao2,bo2,do = Spec_double(cube_obs.data,vel_obs,rvalso,tvalso,vmodo,r_min[k],r_max[k],PA_min,PA_max)
        Summary_obs[k,:] = [ao, bo, co, ao2, bo2, do]
        with open(dirName+'Spec_fit_rmin{:4.2f}_rmax{:4.2f}_params_obs_double.dat'.format(r_min[k],r_max[k]),'w+') as f:
            f.write('Gaussian parameters of a0 b0 c0 a1 d0\n')
            #f.write('%13.6e %13.6e %13.6e %13.6e %13.6e \n'%(ao,bo,co,ao2,do))
            f.write('%13.6e %13.6e %13.6e %13.6e %13.6e %13.6e\n'%(ao,bo,co,ao2,bo2,do))
            f.write('Keplerian corrected spectrum\n')
            for i in range(len(I_aveo)):
                f.write('%13.6e \n'%(I_aveo[i]))
    #for ii in range(cube_obs.header['NAXIS3']): cube_obs.data[ii,:,:] = cube_obs.data[ii,:,:].T
    x3,y3,dy3 = cube_obs.average_spectrum(r_min=r_min[k], r_max=r_max[k], dr=0.1, inc=inc,PA=DPA, mstar=0.63, dist=160.0, x0=dxc, y0=dyc ,PA_min = PA_min, PA_max = PA_max,mask_frame='disk', abs_PA=True)  # Minor axis 1

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
    ax1.plot(cube.velax,I_ave,'r',label='Wind X')
    ax1.plot(x1,y1,'r--')
    if double == 'F': ax1.plot(x1,Gaussian(x1,a,b,c,d),'c-.')
    if double == 'T': ax1.plot(x1,Double(x1,a,b,c,a2,b2,d),'c-.') #b2,c2,
    ax1.plot(cube_wind.velax,I_avew,'b',label='Wind O')
    ax1.plot(x2,y2,'b--')
    if double == 'F': ax1.plot(x1,Gaussian(x1,aw,bw,cw,dw),'m-.')
    if double == 'T': ax1.plot(x1,Double(x1,aw,bw,cw,aw2,bw2,dw),'m-.') #,bw2,cw2
    #ax1.axvline(x=vmod[iy,ix]*1e-3,lw=0.8,ls='--',color='orange')
    #ax1.axvline(x=vmodw[iy,ix]*1e-3,lw=0.8,ls='--',color='g')
    ax1.axvline(x=0.0,lw=0.8,ls='--',color='k')
    ax3 = ax1.twinx()
    ax3.set_xlim(-6,6)
    ax3.set_ylabel('I$_{obs}$ [Jy/beam]')
    ax3.plot(vel_obs ,I_aveo,'k',label='Obs',lw=0.5)
    ax3.plot(x3*1e-3 - 4.5,y3,'k--',lw=0.5)
    ax1.legend(prop={'size':12},loc=0)
    #ax1.text(0.95, 1.05, r'R = {:4.2f} au, $\theta$ = {:4.2f}, V$_C$ ~ {:4.2f} km/s'.format(rvals[iy,ix]*160.0,tvals[iy,ix]*180./np.pi, vmod[iy,ix]*1e-3), ha='right', va='top', transform=ax1.transAxes, color="k",fontsize=13)
    plt.savefig(dirName+'RULup_'+mole+'_rmin{:4.2f}_rmax{:4.2f}_model_spec.pdf'.format(r_min[k],r_max[k]), bbox_inches='tight', pad_inches=0.1)
    #plt.show()
    plt.close()


# =======================================================================================
# Write fitting parameters in summary file
# =======================================================================================
outfile = mole + '_Gaussian_fittings_azi_'+bmaj+'_double_'+double+'_nowind.dat'
np.savetxt(outdir+outfile, Summary_nowind, fmt='%-10.6f')
outfile = mole + '_Gaussian_fittings_azi_'+bmaj+'_double_'+double+'_wind.dat'
np.savetxt(outdir+outfile, Summary_wind, fmt='%-10.6f')
outfile = mole + '_Gaussian_fittings_azi_'+bmaj+'_double_'+double+'_obs.dat'
np.savetxt(outdir+outfile, Summary_obs, fmt='%-10.6f')
