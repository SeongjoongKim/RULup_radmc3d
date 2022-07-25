import numpy as np
import scipy.interpolate as inter
from astropy.io import fits
import scipy.constants as sc
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Rectangle, Circle
from matplotlib.colors import LogNorm
import matplotlib.colors as colors
from astropy.wcs import WCS
from scipy.interpolate import interp1d, griddata
from astropy.convolution import Gaussian2DKernel, convolve
from mpl_toolkits.axes_grid1 import ImageGrid
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredAuxTransformBox
from scipy.stats import binned_statistic
from scipy.optimize import curve_fit
import os

# Some natural constants  ----------------------------------
au  = 1.49598e13     # Astronomical Unit       [cm]
pc  = 3.08572e18     # Parsec                  [cm]
ms  = 1.98892e33     # Solar mass              [g]
ts  = 5.78e3         # Solar temperature       [K]
ls  = 3.8525e33      # Solar luminosity        [erg/s]
rs  = 6.96e10        # Solar radius            [cm]
GG  = 6.67408e-08    # Gravitational constant  [cm^3/g/s^2]
mp  = 1.6726e-24     # Mass of proton          [g]
SB = 5.670374419e-5    # Stefan-Boltzmann constant [ erg cm-2 s-1 K-4]
h = 6.62607004e-27       # Planck function [ cm2 g s-1]
#      Physical constants ----------------------------------
mu = 2.34
kb = 1.38e-16
NA = 6.02e23
mH = 1.0/NA


# Gaussian Fittings # =========================================================
def Gaussian(x,a0,b0,c0,d0):
    return a0*np.exp(-(x-b0)**2/(2*c0**2)) +d0 


def Double(x,a0,b0,c0,a1,b1,c1,d0):
    return a0*np.exp(-(x-b0)**2/(2*c0**2)) + a1*np.exp(-(x-b1)**2/(2*c1**2)) + d0 


def Fit_spectra(x,y,Func,IC,LL,UL):
    try:
        popt, pcov = curve_fit(Func, x, y, p0=IC, bounds=([LL,UL]), absolute_sigma=True)
        return popt, pcov
    except RuntimeError:
        print("Error - curve_fit failed to fit")
    

# Coordinate rotation # =========================================================
def coord_rotate(x,y,z,angle,axis):
    # Rotating the axes
    # inc means rotation angle along x-axis. +z axis will rotate toward +y axis when the standard cartesian coordinates is assumed
    # PA means rotation angle along z-axis. +x axis will rotate toward -y axis when the standard cartesian coordinates is assumed
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
    cc = np.sin(PA*np.pi/180.); dd = np.cos(PA*np.pi/180.)
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

def azi_aver(data, rr, tt, r_min, r_max, PA_min, PA_max):
    r_mask = np.logical_and(rr >= r_min, rr <= r_max)
    PA_mask = np.logical_and(tt >= PA_min*np.pi/180., tt <= PA_max*np.pi/180.)
    mask = r_mask*PA_mask
    n_sample = np.sum(mask) ; idx,idy = np.where(mask >0)
    for k in range(len(idx)):
        I_sample = data[idx[k],idy[k]]
    I_avg = I_sample.mean(); I_std = np.std(I_sample)
    return I_avg, I_std

# Gaussian Convolution # =========================================================
def Gauss_convol(dat,bmaj,bmin,bpa,pixsize_x,pixsize_y,x_size=151,y_size=151):
    # Convolution of V field by a beam size
    gaussian_2D_kernel = Gaussian2DKernel(bmaj/pixsize_x/np.sqrt(8*np.log(2)),bmin/pixsize_y/np.sqrt(8*np.log(2)),bpa/180.*np.pi,x_size=x_size, y_size=y_size)
    # Convolve the image through the 2D beam
    dat_conv = convolve(dat,gaussian_2D_kernel,boundary='extend',normalize_kernel=True)
    return dat_conv

def add_beam(ax,beam,frameon=True):
    (bmaj,bmin,pa)=beam
    box=AnchoredAuxTransformBox(ax.transData, loc=3, frameon=frameon)
    color='k' if frameon else 'w'
    beam_el=Ellipse((0,0),height=bmaj,width=bmin,angle=pa,color=color)
    #beam_el=Ellipse((0,0),height=bmaj,width=bmin,angle=pa,facecolor='w',edgecolor='k',linewidth=0.2)
    box.drawing_area.add_artist(beam_el)
    ax.add_artist(box)
    return

# =====================================================================
class datacube(object):
    """
    Reading data cube fits file and do some interesting analysis and plotting maps & spectra
    """
    # Units setup
    frequency_units = {'GHz': 1e9, 'MHz': 1e6, 'kHz': 1e3, 'Hz': 1e0}
    velocity_units = {'km/s': 1e3, 'm/s': 1e0}
    units = {'V': r'$\mathrm{(km\ s^{-1})}$', 'I': r'$\mathrm{(Jy\ beam^{-1})}$', 'M0': r'$\mathrm{(Jy\ beam^{-1}\ km\ s^{-1})}$', 'M2': r'$\mathrm{(km\ s^{-1})^{2}}$'}
    # Read FITS file
    def __init__(self, path):
        # File name
        self.path = os.path.expanduser(path)
        self.fname = self.path.split('/')[-1]
        
        # FITS data.
        self.header = fits.getheader(path)
        self.data = np.squeeze(fits.getdata(self.path))
        self.data = np.where(np.isfinite(self.data), self.data, 0.0)
        #if self.header['NAXIS'] == 4: self.data = self.data[0,:,:,:]
        #if self.header['NAXIS'] == 3: self.data = self.data

        # Cube properties
        self.nx = self.header['NAXIS1']; self.ny = self.header['NAXIS2']; self.nv = self.header['NAXIS3']
        self.refx = self.header['CRPIX1']; self.refy = self.header['CRPIX2']; self.refv = self.header['CRPIX3']
        if 'BMAJ' in self.header: 
            self.bmaj = self.header['BMAJ']*3.6e3
        else:
            self.bmaj = 0.1
        if 'BMIN' in self.header: 
            self.bmin = self.header['BMIN']*3.6e3
        else:
            self.bmin = 0.1
        if 'BPA' in self.header: 
            self.bpa = self.header['BPA']
        else:
            self.bpa = 0.0
        self.pixsize_x = abs(self.header['CDELT1']*3.6e3); self.pixsize_y = abs(self.header['CDELT2']*3.6e3)
        try: 
            self.bunit = self.header['bunit']
        except KeyError:
            print("No bunit in the header")
        
        # Velocity axis
        if self.header['CTYPE3'] == 'FREQ':
            f0=self.header['CRVAL3']*1e-9
            df=self.header['CDELT3']*1e-9
            restfreq=self.header['RESTFRQ']*1e-9
            c=2.99792458e5 # speed of light (km/s)
            self.v0=-((f0-(self.refv-1)*df)-restfreq)*c/restfreq
            self.dv=-df*c/restfreq
        if self.header['CTYPE3'] == 'VELO-LSR':
            self.dv = self.header['CDELT3']
            self.v0 = self.header['CRVAL3'] - (self.refv-1)*self.dv
        # Set velocity axis of the data cube
        self.velax = np.arange(self.nv)*self.dv + self.v0
        
        
    #  ======== Momemt maps  ========
    def sub_cont(self, nchan=10):
        data_cl = np.zeros((self.ny,self.nx))
        data_cr = np.zeros((self.ny,self.nx))
        for i in range(nchan):
            data_cl += self.data[i,:,:]
            data_cr += self.data[self.nv-1-i,:,:]
        data_cl /= nchan; data_cr /= nchan
        data_c = (data_cl+data_cr)*0.5
        self.data -= data_c[None,:,:]  # continuum subtraction
        return self.data
    
    def collapse_zeroth(self, rms, i_start, i_end):
        dvel = np.diff(self.velax).mean()
        npix = np.sum(self.data != 0.0, axis=0)
        M0 = np.trapz(self.data[i_start:i_end,:,:],dx=dvel,axis=0)
        npix = (i_end-i_start)#*np.ones(M0.shape)
        dM0 = dvel * rms * npix**0.5 * np.ones(M0.shape)
        return M0, dM0
    
    def collapse_first(self, rms, N_cut, i_start, i_end):
        #dvel = np.diff(self.velax).mean()
        data=self.data[i_start:i_end,:,:]
        vpix = self.velax[i_start:i_end, None, None] * np.ones(data.shape)
        weights = 1e-10 * np.random.rand(data.size).reshape(data.shape)
        weights = np.where(data >= N_cut*rms, data, weights)
        M1 = np.average(vpix, weights=weights, axis=0)
        dM1 = (vpix - M1[None, :, :]) * rms / np.sum(weights, axis=0)
        dM1 = np.sqrt(np.sum(dM1**2, axis=0))
        npix = np.sum(data >= N_cut*rms, axis=0)
        M1 = np.where(npix >= 5.0, M1, 0)
        dM1 = np.where(npix >= 5.0, dM1, 0)
        return M1, dM1
    
    def collapse_second(self, rms, N_cut, i_start, i_end):
        #dvel = np.diff(self.velax).mean()
        #data_M1=self.data
        data=self.data[i_start:i_end,:,:]
        vpix = self.velax[i_start:i_end, None, None] * np.ones(data.shape)
        weights = 1e-10 * np.random.rand(data.size).reshape(data.shape)
        weights = np.where(data >= N_cut*rms, data, weights)
        M1 = self.collapse_first(rms, N_cut, i_start, i_end)[0]
        M1 = M1[None, :, :] * np.ones(data.shape)
        M2 = np.sum(weights * (vpix - M1)**2, axis=0) / np.sum(weights, axis=0)
        M2 = np.sqrt(M2)
        dM2 = ((vpix - M1)**2 - M2**2) * rms / np.sum(weights, axis=0)
        dM2 = np.sqrt(np.sum(dM2**2, axis=0)) / 2. / M2
        npix = np.sum(data >= N_cut*rms, axis=0)
        M2 = np.where(npix >= 5.0, M2, 0)
        dM2 = np.where(npix >= 5.0, dM2, 0)
        return M2, dM2
    
    def collapse_eighth(self, rms):
        M8 = np.max(self.data, axis=0)
        dM8 = rms * np.ones(M8.shape)
        return M8, dM8
        
    #  ======== Channel map plotting  ========
    def plot_chmap(self, w, h, nchan=0, lim=2.0, vmax=1.0, vmin=-1.0, cmap="gist_heat", ctext='w', figname='None', save=False):
        data = self.data[nchan:nchan+w*h,:,:]
        beam = ((self.bmaj,self.bmin,self.bpa))
        xlim=[-lim,lim]; ylim=xlim
        xlen=self.nx*self.pixsize_x; ylen=self.ny*self.pixsize_y
        extent=(-xlen/2.,xlen/2.,-ylen/2.,ylen/2.)
        if vmax == 1.0: vmax = data.max()
        if vmin == -1.0: vmin = data.min()
        ### Draw the channel map ###
        fig=plt.figure(figsize=(4*w,2*h))
        grid=ImageGrid(fig,111,nrows_ncols=(h,w),axes_pad=0.,label_mode='1', share_all=True, cbar_location='right', cbar_mode='single',cbar_size='3%',cbar_pad='1%')
        for i,ax in enumerate(grid):
            im=ax.imshow(data[i],extent=extent,origin='lower',vmax=vmax,vmin=vmin,cmap=cmap)
            #if mask != None: ax.contour(mask,[1],linestyles='solid',colors='b')
            txt='%.2f km/s' % self.velax[nchan+i]
            ax.text(0.95, 0.95, txt, ha='right', va='top', transform=ax.transAxes, color=ctext)
            ax.set_xlim(xlim); ax.set_ylim(ylim)
            if i==w*(h-1):
                ax.set_xlabel(r"$\Delta\alpha\;(\mathrm{arcsec})$")
                ax.set_ylabel(r"$\Delta\delta\;(\mathrm{arcsec})$")
                add_beam(ax,beam,frameon=False)
        grid.cbar_axes[0].colorbar(im)
        grid.cbar_axes[0].set_ylabel(self.units['V'])
        #grid.cbar_axes[0].set_yticks((vmin,vmax))
        if save:
            plt.savefig(figname, bbox_inches='tight', pad_inches=0.1, dpi=150)
        else:
            plt.show()
            
            
    #  ======== 2D map plotting  ========
    def plot_2Dmap(self, dat, inc, dpa, d_pc, vmin, vmax, scale='linear', cmap='jet', unit='I', R1=0.0, RC1='k', R2=0.0, RC2='k', figname='None', save=False):
        wcs = WCS(self.header).slice([self.nx,self.ny])                         # calculate RA, Dec
        if R1 > 0.0: C1 = Ellipse( (int(self.nx/2), int(self.ny/2)), R1/d_pc*100, R1/d_pc*100*np.cos(inc*np.pi/180.), angle=180.-dpa, edgecolor=RC1, facecolor='none',ls='--')  # Keplerian disk
        if R2 > 0.0: C2 = Ellipse( (int(self.nx/2), int(self.ny/2)), R2/d_pc*100, R2/d_pc*100*np.cos(inc*np.pi/180.), angle=180.-dpa, edgecolor=RC2, facecolor='none',ls='--')   # Outer envelope boundary
        fig=plt.figure(num=None, figsize=(8,6), facecolor='w', edgecolor='k')
        ax = fig.add_subplot(projection=wcs)
        if scale == 'linear': norm = colors.Normalize(vmin=vmin,vmax=vmax)
        if scale == 'log': norm = colors.LogNorm(vmin=vmin,vmax=vmax)
        im=ax.imshow(dat,origin='lower',cmap=cmap,norm=norm)
        #TT = ax.contour(vmod,[-0.75,-0.5,-0.3,-0.1,0.0,0.1,0.3,0.5,0.75],linestyles='solid',colors='gray')
        #plt.clabel(TT)
        ax.set_xlabel('RA',fontsize=15)
        ax.set_ylabel('Dec',fontsize=15)
        if R1 > 0.0: ax.add_patch(C1)
        if R2 > 0.0: ax.add_patch(C2)
        #ax.text(0.95, 0.95, freq[i], ha='right', va='top', transform=ax.transAxes, color="w",fontsize=15)
        ax.tick_params(axis='both',which='both',length=4,width=1,labelsize=12,direction='in',color='w')
        #ax.margins(x=-0.375,y=-0.375)
        cbar=plt.colorbar(im, shrink=0.9)
        cbar.set_label(self.units[unit], size=10)
        if save:
            plt.savefig(figname, bbox_inches='tight', pad_inches=0.1, dpi=150)
        else:
            plt.show()
        plt.clf()

    #  ======== projected disks  ========
    def get_disk_coord(self, inc, dpa, z0=0.0, psi=1.0, z1=0.0, phi=1.0, rt0=np.inf, qt0=1.0):
        # Set sky plane xy # =========================================================
        x_sky,y_sky = self.get_sky_coord()
        x_sky,y_sky = np.ravel(x_sky), np.ravel(y_sky)

        # Rotated axes direction vectors
        ux_sky, uy_sky, uz_sky = rotated_axis(inc, dpa)
        #print(ux_sky, uy_sky, uz_sky)
        # Axes rotation from sky plane to disk plane
        ux_disk, uy_disk, uz_disk = np.linalg.inv(np.array((ux_sky,uy_sky,uz_sky)))
        #print(ux_disk, uy_disk, uz_disk)

        # Set disk coordinate vector in disk # ==========================================
        x_disk, y_disk = 2.*x_sky, 2.*y_sky
        r_disk = np.hypot(x_disk,y_disk)
        z_disk = ( z0 * np.power(r_disk, psi) + z1 * np.power(r_disk, phi) ) *np.exp( -np.power(r_disk/rt0, qt0) ) 
        #z_disk = ( z0 * (1 + np.power(r_disk, psi)) + z1 * (1 + np.power(r_disk, phi)) ) *np.exp( -np.power(r_disk/rt0, qt0) ) 
        
        # Calculate the projected (x', y', z') & (vx', vy', vz') in sky plane # ==============================
        xx_sky = vector_proj(x_disk, y_disk, z_disk, ux_sky)
        yy_sky = vector_proj(x_disk, y_disk, z_disk, uy_sky)
        zz_sky = vector_proj(x_disk, y_disk, z_disk, uz_sky)
        
        # Interpolate z_sky # ================================================================================
        z_sky = griddata((xx_sky,yy_sky),zz_sky,(x_sky,y_sky),method='linear')
        # Deproject observed pixels on the disk plane # ======================================================
        xx_disk = vector_proj(x_sky, y_sky, z_sky, ux_disk)
        yy_disk = vector_proj(x_sky, y_sky, z_sky, uy_disk)
        zz_disk = vector_proj(x_sky, y_sky, z_sky, uz_disk)        
        return xx_disk.reshape((self.nx,self.ny)), yy_disk.reshape((self.nx,self.ny)), zz_disk.reshape((self.nx,self.ny))
        
    def get_sky_coord(self):
        # Set sky plane xy # =========================================================
        x=np.arange(-self.nx/2.,self.nx/2.)*self.pixsize_x#*d_pc # in au
        y=np.arange(-self.ny/2.,self.ny/2.)*self.pixsize_y#*d_pc # in au
        qq = np.meshgrid(x,y)
        return qq[0], qq[1]
    
    def get_disk_rtheta(self, inc, dpa, z0=0.0, psi=1.0, z1=0.0, phi=1.0, rt0=np.inf, qt0=1.0):
        xx_disk, yy_disk, zz_disk = self.get_disk_coord(inc, dpa,z0, psi, z1, phi, rt0, qt0)
        r_disk = np.hypot(xx_disk, yy_disk)
        t_disk = np.arctan2(yy_disk,xx_disk)
        return r_disk, t_disk        
    
    def get_mask(self,r,t,rin,rout,PAmin,PAmax):
        r_mask= np.logical_and(r>=rin, r<=rout)
        t_mask = np.logical_and(t>=np.radians(PAmin),t<=np.radians(PAmax))
        mask = r_mask*t_mask#; npnts = np.sum(mask)
        return mask

    def get_vkep(self, inc, dpa, mstar=1.0, d_pc=140.0, z0=0.0, psi=1.0, z1=0.0, phi=1.0, rt0=np.inf, qt0=1.0):
        x, y ,z = self.get_disk_coord(inc, dpa, z0=z0, psi=psi, z1=z1, phi=phi, rt0=rt0, qt0=qt0)
        r = np.hypot(x,y)
        vkep = GG*mstar*np.power(r*d_pc*au,2.0) 
        vkep = np.sqrt(vkep/np.power(np.hypot(r*d_pc*au,z*d_pc*au),3.0))
        vkep = np.where(np.isfinite(vkep), vkep, 0.0) # in cm/s unit
        theta = np.arctan2(y,x)
        vvx = vkep*np.sin(theta); vvy = vkep*np.cos(theta); vvz = np.zeros_like(vvx)
        ux_sky, uy_sky, uz_sky = rotated_axis(inc, dpa)
        vvz_sky = vector_proj(vvx, vvy, vvz, uz_sky) # Final velocity field on the sky plane
        vvz_sky = np.where(np.isfinite(vvz_sky), vvz_sky, 0.0)      # Check Nan and +/-inf and replace them to 0
        return vvz_sky
                
    def get_vlos_spec(self, vmodel, mask):
        npnts = np.sum(mask); k=0
        vl = np.zeros(npnts); inten = np.zeros((npnts,self.nv))
        for i in range(self.nx):
            for j in range(self.ny):
                if mask[i,j]:
                    vl[k] = vmodel[i,j]
                    inten[k,:] = self.data[:,j,i]
                    k += 1
        return vl, inten

    def shift_spec(self, rr, tt, vmodel, r_min, r_max, PA_min, PA_max):
        mask = self.get_mask(rr, tt, r_min, r_max, PA_min, PA_max)
        ndpnts = np.sum(mask)
        shifted = np.zeros(self.nv)
        ndata = np.ones(self.nv)*ndpnts
        for i in range(self.nx):
            for j in range(self.ny):
                if mask[j,i]:
                    shifted_spec = interp1d(self.velax - vmodel[i, j], self.data[:, j, i], bounds_error=False)(self.velax)
                    ndata[np.isnan(shifted_spec)] -= 1
                    shifted_spec[np.isnan(shifted_spec)] = 0.0
                    shifted += shifted_spec
        shifted /= ndata
        return shifted
    
    def get_azi_aver_spec(self, vref, inc, dpa, vmod, rin, rout, PA_min=-180.0, PA_max=180.0, 
                          z0=0.0, psi=1.0, z1=0.0, phi=1.0, rt0=np.inf, qt0=1.0):
        r, theta = self.get_disk_rtheta(inc, dpa, z0=z0, psi=psi, z1=z1, phi=phi, rt0=rt0, qt0=qt0)
        mask = self.get_mask(r,theta,rin,rout,PA_min,PA_max)
        vlos, spectrum = self.get_vlos_spec(vmod, mask)
        vpnts = np.ravel(self.velax[None,:]-vlos[:,None]); spnts = np.ravel(spectrum)
        idv = np.argsort(vpnts)
        vsort = vpnts[idv]; ssort = spnts[idv];
        y = binned_statistic(vsort, ssort, statistic='mean', bins=vref)[0]
        aver_spec = np.where(np.isfinite(y), y, 0.0)
        std_spec = binned_statistic(vsort, ssort, statistic='std', bins=vref)[0]
        return aver_spec, std_spec
        
    def get_teardrop(self, inc, dpa, vmod, rin, rout, nr, PA_min=-180.0, PA_max=180.0, z0=0.0, psi=1.0, z1=0.0, phi=1.0, rt0=np.inf, qt0=1.0):
        r_bin, rc = self.get_rbin(rin, rout, nr)
        # Set velocity axis
        vref = (np.arange(self.nv+1)-0.5)*self.dv + self.v0
        vcent = np.average([vref[1:], vref[:-1]], axis=0)
        aver_spec = np.zeros((nr,self.nv)); std_spec = np.zeros((nr,self.nv))
        for i in range(nr):
            r_in = r_bin[i]; r_out = r_bin[i+1]
            aver_spec[i,:], std_spec[i,:] = self.get_azi_aver_spec(vref, inc, dpa, vmod, rin, rout, PA_min=-180.0, PA_max=180.0, 
                                  z0=0.0, psi=1.0, z1=0.0, phi=1.0, rt0=np.inf, qt0=1.0)
        return vcent, aver_spec, std_spec
        
    def get_rbin(self, rin, rout, nr):
        dr = (rout - rin)/nr
        r_bin = np.arange(rin,rout+dr,dr)
        rc = (r_bin[1:nr+1] + r_bin[0:nr])*0.5
        return r_bin, rc




