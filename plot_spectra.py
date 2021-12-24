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
# Parameter modification
# =======================================================================================
# Masking parameters are modified in lines 82-103, Parameter setup part
# Target file names are modified in lines 105-125, Filename setup part
# Output directory is modified in lines 160-168, the top of fitting and plotting part

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
r0 = 0.05;  r1 = 0.85; dr =0.1
r_min = np.arange(r0,r1,dr); r_max = r_min + dr
DPA = 121.0; PA_min = 45; PA_max = 135
dxc = 0.00; dyc = 0.00
z0 = 0.0;psi = 1.0; z1 = 0.0; phi = 1.0

n_gauss = 4
if double == 'T': n_gauss = 5
Summary_nowind = np.zeros((len(r_min), n_gauss))
Summary_wind = np.zeros((len(r_min),n_gauss))
Summary_obs = np.zeros((len(r_min),n_gauss))

# =======================================================================================
# File name setup
# =======================================================================================
# Set No wind model fits file
fdir = '/Users/kimsj/Documents/RADMC-3D/radmc3d-2.0/RU_Lup_test/Automatics/Fin_script/fiducial/'
fitsname = 'RULup_'+mole+'_fiducial_'+bmaj+'.fits'  #
cube = imagecube(fdir+fitsname)
#for ii in range(cube.header['NAXIS3']): cube.data[ii,:,:] = cube.data[ii,:,:].T

# Set Wind model fits file
fdir2 = '/Users/kimsj/Documents/RADMC-3D/radmc3d-2.0/RU_Lup_test/Automatics/Fin_script/fiducial_wind/'
windname = 'RULup_'+mole+'_fiducial_wind_CN0.35_Cw1e-4_'+bmaj+'.fits'#
cube_wind = imagecube(fdir2+windname)
#for ii in range(cube_wind.header['NAXIS3']): cube_wind.data[ii,:,:] = cube_wind.data[ii,:,:].T

# Set Observation fits file
obsname = '/Users/kimsj/Documents/RU_Lup/Fin_fits/'+mole+'_selfcal_wc_matched_cube500.fits'
cube_obs = imagecube(obsname)
nv_obs = cube_obs.header['NAXIS3']
#for ii in range(cube_obs.header['NAXIS3']): cube_obs.data[ii,:,:] = cube_obs.data[ii,:,:].T
vel_obs = cube_obs.velax*1e-3 - 5.0 #0.52 +np.arange(nv_obs)*0.084 - 5.0

# Output directory of Gaussian fitting parameter summary
outdir = '../spec_figure/CN0.35_Cw1e-4/'

# =======================================================================================
# Making fittings of spectra and plotting
# =======================================================================================
for k in range(len(r_min)):
    dirName = '../spec_figure/CN0.35_Cw1e-4/'+mole+'_r{:4.2f}-{:4.2f}_PA{:03d}-{:03d}_'.format(r_min[k],r_max[k],PA_min,PA_max)+bmaj+'/'  # Output directory set
    if double == 'T':
        dirName = '../spec_figure/CN0.35_Cw1e-4/'+mole+'_r{:4.2f}-{:4.2f}_PA{:03d}-{:03d}_'.format(r_min[k],r_max[k],PA_min,PA_max)+bmaj+'_double/'
    if not os.path.exists(dirName):
        os.mkdir(dirName)
        print("Directory " , dirName ,  " Created ")
    else:
        print("Directory " , dirName ,  " already exists")
    # =======================================================================================
    # Plotting the masking regions and spectra
    # =======================================================================================
    #shifted_cube = cube.shifted_cube(r_min=r_min,r_max=r_max[k],inc=inc,PA=DPA, mstar=0.63, dist=160.0, x0=dxc, y0=dyc)
    x1,y1,dy1 = cube.average_spectrum(r_min=r_min[k], r_max=r_max[k], dr=0.1, inc=inc,PA=DPA, mstar=0.63, dist=160.0, x0=dxc, y0=dyc ,PA_min = PA_min, PA_max = PA_max,mask_frame='disk', abs_PA=True)  # Minor axis 1
    fig, ax = plt.subplots()
    ax.imshow(cube.data[100,:,:], origin='lower',extent=cube.extent, vmin=cube.data.min(), vmax=cube.data[100,:,:].max()*0.95)
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
        a,b,c,d,idx,idy = Spec_fit(cube.data,cube.velax,rvals,tvals,vmod,r_min[k],r_max[k],PA_min,PA_max)
        Summary_nowind[k,:] = [a.mean(), b.mean(), c.mean(), d.mean()]
        with open(dirName+'Spec_fit_rmin{:4.2f}_rmax{:4.2f}_params_nowind.dat'.format(r_min[k],r_max[k]),'w+') as f:
            f.write('Averages of a b c d\n')
            f.write('%13.6e %13.6e %13.6e %13.6e\n'%(a.mean(),b.mean(),c.mean(),d.mean()))
            f.write('a b c d idx idy\n')
            for i in range(len(a)):
                f.write('%13.6e %13.6e %13.6e %13.6e %4d %4d\n'%(a[i],b[i],c[i],d[i],idx[i],idy[i]))
    if double == 'T':
        a,b,c,a2,d,idx,idy = Spec_double(cube.data,cube.velax,rvals,tvals,vmod,r_min[k],r_max[k],PA_min,PA_max)
        Summary_nowind[k,:] = [a.mean(), b.mean(), c.mean(), a2.mean(), d.mean()]
        with open(dirName+'Spec_fit_rmin{:4.2f}_rmax{:4.2f}_params_nowind_double.dat'.format(r_min[k],r_max[k]),'w+') as f:
            f.write('Averages of a0 b0 c0 a1 d0\n')
            f.write('%13.6e %13.6e %13.6e %13.6e %13.6e \n'%(a.mean(),b.mean(),c.mean(),a2.mean(),d.mean()))
            f.write('a0 b0 c0 a1 d0 idx idy\n')
            for i in range(len(a)):
                f.write('%13.6e %13.6e %13.6e %13.6e %13.6e %4d %4d\n'%(a[i],b[i],c[i],a2[i],d[i],idx[i],idy[i]))

    # =======================================================================================
    # Wind model
    # =======================================================================================
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
        aw,bw,cw,dw,idxw,idyw = Spec_fit(cube_wind.data,cube_wind.velax,rvalsw,tvalsw,vmodw,r_min[k],r_max[k],PA_min,PA_max)
        Summary_wind[k,:] = [aw.mean(), bw.mean(), cw.mean(), dw.mean()]
        with open(dirName+'Spec_fit_rmin{:4.2f}_rmax{:4.2f}_params_wind.dat'.format(r_min[k],r_max[k]),'w+') as f:
            f.write('Averages of a b c d\n')
            f.write('%13.6e %13.6e %13.6e %13.6e\n'%(aw.mean(),bw.mean(),cw.mean(),dw.mean()))
            f.write('a b c d idx idy\n')
            for i in range(len(aw)):
                f.write('%13.6e %13.6e %13.6e %13.6e %4d %4d\n'%(aw[i],bw[i],cw[i],dw[i],idxw[i],idyw[i]))
    if double == 'T':
        aw,bw,cw,aw2,dw,idxw,idyw = Spec_double(cube_wind.data,cube_wind.velax,rvalsw,tvalsw,vmodw,r_min[k],r_max[k],PA_min,PA_max)
        Summary_wind[k,:] = [aw.mean(), bw.mean(), cw.mean(), aw2.mean(), dw.mean()]
        with open(dirName+'Spec_fit_rmin{:4.2f}_rmax{:4.2f}_params_wind_double.dat'.format(r_min[k],r_max[k]),'w+') as f:
            f.write('Averages of a0 b0 c0 a1 d0\n')
            f.write('%13.6e %13.6e %13.6e %13.6e %13.6e \n'%(aw.mean(),bw.mean(),cw.mean(),aw2.mean(),dw.mean()))
            f.write('a0 b0 c0 a1 d0 idx idy\n')
            for i in range(len(aw)):
                f.write('%13.6e %13.6e %13.6e %13.6e %13.6e %4d %4d\n'%(aw[i],bw[i],cw[i],aw2[i],dw[i],idxw[i],idyw[i]))

    # =======================================================================================
    # Observation
    # =======================================================================================
    #x3,y3,dy3 = cube_obs.average_spectrum(r_min=r_min[k], r_max=r_max[k], dr=0.1, inc=inc,PA=DPA, mstar=0.63, dist=160.0, x0=dxc, y0=dyc ,PA_min = PA_min, PA_max = PA_max,mask_frame='disk', abs_PA=True)  # Minor axis 1
    if bmaj == 'bmaj51':
        rvalso, tvalso, _ = cube_obs.disk_coords(x0=dxc,y0=dyc,inc=inc,PA=DPA,z0=z0,psi=psi,z1=z1,phi=phi)
        vmodo = cube_obs._keplerian(rpnts=rvalso,mstar=0.63,dist=160.0, inc=inc)
        vmodo *= np.cos(tvalso)
        b_maj = cube_obs.header['BMAJ']*3.6e3; b_min = cube_obs.header['BMIN']*3.6e3; bpa = cube_obs.header['BPA']
        pixsize_x = abs(cube_obs.header['CDELT1']*3.6e3); pixsize_y = abs(cube_obs.header['CDELT2']*3.6e3)
        gaussian_2D_kernel = Gaussian2DKernel(b_maj/pixsize_x/np.sqrt(8*np.log(2)),b_min/pixsize_y/np.sqrt(8*np.log(2)),bpa/180.*np.pi,x_size=151, y_size=151)
        # Convolve the image through the 2D beam
        vmodo = convolve(vmodo,gaussian_2D_kernel,boundary='fill',fill_value='0',normalize_kernel=True)
        if double == 'F':
            ao,bo,co,do,idxo,idyo = Spec_fit(cube_obs.data,cube_obs.velax,rvalso,tvalso,vmodo,r_min[k],r_max[k],PA_min,PA_max)
            Summary_obs[k,:] = [ao.mean(), bo.mean(), co.mean(), do.mean()]
            with open(dirName+'Spec_fit_rmin{:4.2f}_rmax{:4.2f}_params_obs.dat'.format(r_min[k],r_max[k]),'w+') as f:
                f.write('Averages of a b c d\n')
                f.write('%13.6e %13.6e %13.6e %13.6e\n'%(ao.mean(),bo.mean(),co.mean(),do.mean()))
                f.write('a b c d idx idy\n')
                for i in range(len(ao)):
                    f.write('%13.6e %13.6e %13.6e %13.6e %4d %4d\n'%(ao[i],bo[i],co[i],do[i],idxo[i],idyo[i]))
        if double == 'T':
            ao,bo,co,ao2,do,idxo,idyo = Spec_double(cube_obs.data,cube_obs.velax,rvalso,tvalso,vmodo,r_min[k],r_max[k],PA_min,PA_max)
            Summary_obs[k,:] = [ao.mean(), bo.mean(), co.mean(), ao2.mean(), do.mean()]
            with open(dirName+'Spec_fit_rmin{:4.2f}_rmax{:4.2f}_params_obs_double.dat'.format(r_min[k],r_max[k]),'w+') as f:
                f.write('Averages of a0 b0 c0 a1 d0\n')
                f.write('%13.6e %13.6e %13.6e %13.6e %13.6e \n'%(ao.mean(),bo.mean(),co.mean(),ao2.mean(),do.mean()))
                f.write('a0 b0 c0 a1 d0 idx idy\n')
                for i in range(len(ao)):
                    f.write('%13.6e %13.6e %13.6e %13.6e %13.6e %4d %4d\n'%(ao[i],bo[i],co[i],ao2[i],do[i],idxo[i],idyo[i]))

    # =======================================================================================
    # Plotting the spectrum at each pixels included in the mask
    # =======================================================================================
    del_x = idx; del_y = idy
    for i in range(100): #len(del_x)):
        ix = del_x[i];  iy=del_y[i]
        spec = cube.data[:,iy,ix]
        #shifted_spec = interp1d(x1-vmod[iy,ix]*1e-3,spec,bounds_error=False)(x1)
        spec_wind = cube_wind.data[:,iy,ix]
        #shifted_wind = interp1d(x1-vmod[iy,ix]*1e-3,spec_wind,bounds_error=False)(x1)
        spec_obs = cube_obs.data[:,iy,ix]
        fig = plt.figure(figsize=(12,5))
        grdspec = gridspec.GridSpec(ncols=2, nrows=1, figure=fig)
        ax1 = fig.add_subplot(grdspec[0])
        ax1.set_xlim(-6,6)
        ax1.set_xlabel('Vel [km/s]')
        ax1.set_ylabel('I$_{model}$ [Jy/beam]')
        ax1.plot(x1,spec,'r',label='Wind X')
        #ax1.plot(x1,shifted_spec,'r--')
        if double == 'F': ax1.plot(x1,Gaussian(x1,a[i],b[i],c[i],d[i]),'c-.')
        if double == 'T': ax1.plot(x1,Double(x1,a[i],b[i],c[i],a2[i],d[i]),'c-.')
        ax1.plot(x1,spec_wind,'b',label='Wind O')
        #ax1.plot(x1,shifted_wind,'b--')
        if double == 'F': ax1.plot(x1,Gaussian(x1,aw[i],bw[i],cw[i],dw[i]),'m-.')
        if double == 'T': ax1.plot(x1,Double(x1,aw[i],bw[i],cw[i],aw2[i],dw[i]),'m-.')
        ax1.axvline(x=vmod[iy,ix]*1e-3,lw=0.8,ls='--',color='g')
        ax1.axvline(x=0.0,lw=0.8,ls='--',color='k')
        ax3 = ax1.twinx()
        ax3.set_xlim(-6,6)
        ax3.set_ylabel('I$_{obs}$ [Jy/beam]')
        ax3.plot(vel_obs,spec_obs,'k--',label='Obs',lw=0.3)
        if bmaj == 'bmaj51':
            if double == 'F': ax3.plot(vel_obs,Gaussian(vel_obs,ao[i],bo[i],co[i],do[i]),'k-.', lw=0.5)
            if double == 'T': ax3.plot(vel_obs,Double(vel_obs,ao[i],bo[i],co[i],ao2[i],do[i]),'k-.', lw=0.5)
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


# =======================================================================================
# Write fitting parameters in summary file
# =======================================================================================
outfile = mole + '_Gaussian_fittings_pix_'+bmaj+'_double_'+double+'_nowind.dat'
np.savetxt(outdir+outfile, Summary_nowind, fmt='%-10.6f')
outfile = mole + '_Gaussian_fittings_pix_'+bmaj+'_double_'+double+'_wind.dat'
np.savetxt(outdir+outfile, Summary_wind, fmt='%-10.6f')
outfile = mole + '_Gaussian_fittings_pix_'+bmaj+'_double_'+double+'_obs.dat'
np.savetxt(outdir+outfile, Summary_obs, fmt='%-10.6f')
