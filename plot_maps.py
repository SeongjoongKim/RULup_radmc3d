from matplotlib.colors import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.patches import Ellipse, Rectangle, Circle
from matplotlib.colors import LogNorm
from astropy.utils.data import get_pkg_data_filename
from astropy.wcs import WCS
from matplotlib.offsetbox import (AnchoredOffsetbox, AuxTransformBox, DrawingArea, TextArea, VPacker)
import glob
from astropy.io import fits
from astropy import units as u
#from scipy import stats
#import bettermoments.collapse_cube as bm
#from Model_setup_subroutines import *
import argparse

# Constants
c = 2.99792458e5 # [km/s]
au  = 1.49598e13     # Astronomical Unit       [cm]
pc  = 3.08572e18     # Parsec                  [cm]

# Parser the arguments
parser = argparse.ArgumentParser()
parser.add_argument('-file', default='None', type=str, help='Input test title')
parser.add_argument('-dist', default=140.0, type=float, help='Target distance in parsec unit. Default is 140 pc')
args = parser.parse_args()
test = args.file
d_pc = args.dist

# Species for the mapping
species = ['12CO_2-1','13CO_2-1','13CO_3-2','C18O_2-1','C18O_3-2','CN_3-2','cont220GHz','cont336GHz']
freq = ['12CO_2-1','13CO_2-1','13CO_3-2','C18O_2-1','C18O_3-2','CN_3-2','220 GHz','336 GHz']
#species = ['12CO_2-1','cont336GHz']
#freq = ['12CO_2-1','336 GHz']
#vmax = [0.3,0.3]
#vmin = [1e-3,1e-3]

inc=25.0 ; dpa=59
x0_cont_peak=250 ; y0_cont_peak=250

for i in range(0,len(species)):
    filename = glob.glob('./RULup_'+species[i]+'_'+test+'.fits')[0]  # find the target fits file
    hdu = fits.open(filename)[0]   # Read the fits file: header + data
    data = hdu.data#[0,0,:,:]      # Save the data part
    nx = hdu.header['NAXIS1']; ny = hdu.header['NAXIS2']; nv = hdu.header['NAXIS3']  # Set up axis lengths
    if len(data.shape) == 3: data = data.reshape(-1,nv,ny,nx)    # Match the data shape to (1,nv, ny, nx)
    wcs = WCS(hdu.header).slice([nx,ny])                         # calculate RA, Dec
    bmaj=hdu.header['BMAJ']; bmin=hdu.header['BMIN']; bpa=hdu.header['BPA'] # Read beam parameter
    bmaj=(bmaj*u.degree).to(u.arcsec).value; bmin=(bmin*u.degree).to(u.arcsec).value # Convert from degree to arcsec
    cell=abs(hdu.header['CDELT2']); Cell=(cell*u.degree).to(u.arcsec).value # Read pixel size in arcsec
    beam = Ellipse((50., 50.), bmaj*100, bmin*100, angle=90.+bpa, edgecolor='white', facecolor='white')
    disk = Ellipse((x0_cont_peak, y0_cont_peak), 240./150.*100, 240./150.*100*np.cos(inc*np.pi/180.), angle=90.-dpa, edgecolor='white', facecolor='none',ls='--')  # Keplerian disk
    ring = Ellipse((x0_cont_peak, y0_cont_peak), 520./150.*100, 520./150.*100*np.cos(inc*np.pi/180.), angle=90.-dpa, edgecolor='cyan', facecolor='none',ls='--')   # Outer envelope boundary
    # Making moment 0 map for the line emissions -----------------------------------------------
    data = np.swapaxes(data, 0, 3); data = np.swapaxes(data, 1, 2) # reshape the data in (nx,ny,nv,1)
    #print(data.max(),data.min())
    chan = hdu.header['CDELT3']   # delta_V in km/s
    if (nv>=2):
        M0 = np.trapz(data[:,:,:,0], dx=chan, axis=2)
    else:
        M0 = data[:,:,0,0]
    # Plotting ----------------------------------------------------------------------------
    fig=plt.figure(num=None, figsize=(8,6), dpi=100, facecolor='w', edgecolor='k')
    ax = fig.add_subplot(projection=wcs)
    im=ax.imshow(M0,origin='lower',vmax=M0.max(),vmin=M0.max()*1e-2,cmap="gist_heat")
    ax.set_xlabel('RA',fontsize=15)
    ax.set_ylabel('Dec',fontsize=15)
    ax.text(0.95, 0.95, freq[i], ha='right', va='top', transform=ax.transAxes, color="w",fontsize=15)
    ax.tick_params(axis='both',which='both',length=4,width=1,labelsize=12,direction='in',color='w')
    ax.add_patch(beam)
    ax.add_patch(disk)
    ax.add_patch(ring)
    #ax.margins(x=-0.375,y=-0.375)
    cbar=plt.colorbar(im, shrink=0.9)
    if (nv>=2):
        cbar.set_label(r'$\mathrm{Jy\ Beam^{-1}}\ km\ s^{-1}$', size=10)
    else:
        cbar.set_label(r'$\mathrm{Jy\ Beam^{-1}}$', size=10)
    plt.savefig('RULup_'+species[i]+'_'+test+'_maps.pdf', bbox_inches='tight', pad_inches=0.1,dpi=100)
    M0 = M0.reshape(nx,ny,-1)
    M0 = M0.reshape(nx,ny,1,-1)
    M0 = np.swapaxes(M0, 0, 3); M0 = np.swapaxes(M0, 1, 2)
    # Create M0 fits file
    if (nv>=2):
        # Writing the new fits header
        hdr = fits.Header()
        hdr['BSCALE'] = 1.e0
        hdr['BZERO'] = 0.e0
        hdr['BMAJ'] = bmaj*0.000277778 # Convert to degree
        hdr['BMIN'] = bmin*0.000277778 # Convert to degree
        hdr['BPA'] = bpa
        hdr['BTYPE'] = 'Intensity'
        hdr['BUNIT'] = 'Jy/beam'
        hdr['RADESYS'] = 'ICRS'
        hdr['LONPOLE'] = 1.80e2
        hdr['LATPOLT'] = -3.782110684167E+01
        hdr['SIMPLE'] = 'T'
        hdr['BITPIX'] = 32
        hdr['NAXIS'] = 4
        hdr['NAXIS1'] = hdu.header['NAXIS1']
        hdr['NAXIS2'] = hdu.header['NAXIS2']
        hdr['NAXIS3'] = 1
        hdr['NAXIS4'] = 1
        hdr['CDELT1'] = hdu.header['CDELT1']
        hdr['CRPIX1'] = hdu.header['CRPIX1']
        hdr['CRVAL1'] = hdu.header['CRVAL1']
        hdr['CTYPE1'] = 'RA---SIN'
        hdr['CUNIT1'] = 'deg'
        hdr['CDELT2'] = hdu.header['CDELT2']
        hdr['CRPIX2'] = hdu.header['CRPIX2']
        hdr['CRVAL2'] = hdu.header['CRVAL2']
        hdr['CTYPE2'] = 'DEC--SIN'
        hdr['CUNIT2'] = 'deg'
        hdr['CTYPE3'] = 'VELO-LSR'
        hdr['CDELT3'] = 0.
        hdr['CRPIX3'] = 1.
        hdr['CRVAL3'] = 0.
        hdr['CTYPE4'] = 'STOKES'
        hdr['CDELT4'] = 1.
        hdr['CRPIX4'] = 1.
        hdr['CRVAL4'] = 1.
        hdr['EPOCH'] = 2000.
        hdr['RESTFREQ'] = hdu.header['RESTFREQ']
        hdr['OBJECT'] = 'RU_Lup'
        hdu = fits.PrimaryHDU(M0,hdr)
        hdu.writeto('RULup_'+species[i]+'_'+test+'_M0.fits',overwrite=True,output_verify='fix')
        



#if save:
#    plt.savefig(savename, bbox_inches='tight', pad_inches=0.1)
#else:
#    plt.show()
