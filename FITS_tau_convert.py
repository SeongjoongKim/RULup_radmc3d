import numpy as np
#import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from astropy.convolution import Gaussian2DKernel, convolve
from astropy.io import fits
from datetime import datetime
from Model_setup_subroutines import *
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-file', default='None', type=str, help='input test title')
parser.add_argument('-bmaj', default=0.05, type=float, help='Beam major axis in arcsec unit. Default is 0.05 arcsec')
parser.add_argument('-bmin', default=0.05, type=float, help='Beam minor axis in arcsec unit. Default is 0.05 arcsec')
parser.add_argument('-bpa', default=0.0, type=float, help='Beam position angle in degree unit. Default is 0.0 degree')
parser.add_argument('-dist', default=140.0, type=float, help='Target distance in parsec unit. Default is 140 pc')
args = parser.parse_args()

# Constants
c = 2.99792458e5 # [km/s]
au  = 1.49598e13     # Astronomical Unit       [cm]
pc  = 3.08572e18     # Parsec                  [cm]

# Setup argument varialbles
d_pc = args.dist
test = args.file
bmaj = args.bmaj
bmin = args.bmin
bpa = args.bpa
outbmaj = str(int(bmaj*100))

# input and output filenames
fnameread      = ['image_12CO_21_tau1.out','image_13CO_21_tau1.out','image_13CO_32_tau1.out','image_C18O_21_tau1.out','image_C18O_32_tau1.out','image_CN_32_tau1.out','image_cont220GHz_tau1.out','image_cont336GHz_tau1.out']
outfile=['RULup_12CO_2-1_'+test+'_bmaj'+str(outbmaj)+'_tau1.fits','RULup_13CO_2-1_'+test+'_bmaj'+str(outbmaj)+'_tau1.fits','RULup_13CO_3-2_'+test+'_bmaj'+str(outbmaj)+'_tau1.fits','RULup_C18O_2-1_'+test+'_bmaj'+str(outbmaj)+'_tau1.fits','RULup_C18O_3-2_'+test+'_bmaj'+str(outbmaj)+'_tau1.fits','RULup_CN_3-2_'+test+'_bmaj'+str(outbmaj)+'_tau1.fits','RULup_cont220GHz_'+test+'_bmaj'+str(outbmaj)+'_tau1.fits','RULup_cont336GHz_'+test+'_bmaj'+str(outbmaj)+'_tau1.fits']
freq0=[230.538,220.398684,330.587965,219.9762,329.330553,340.247770,219.9762,336.2768]

#fnameread      = ['image_12CO_21.out','image_cont336GHz.out']
#outfile=['RULup_12CO_2-1_'+test+'_bmaj'+outbmaj+'.fits','RULup_cont336GHz_'+test+'_bmaj'+outbmaj+'.fits']
#freq0=[230.538,336.2768]

# observation fits file for copying header information
B6dir = '/lfs09/kimso/RU_Lup/Huang_data/new_calibrated/'
filename = B6dir+'13CO_all_selfcal_p1st_matched_cube500.fits'
hdu = fits.open(filename)[0]
#header_new = hdu.header

for j in range(len(fnameread)):
    # Read the input image ascii file
    now = datetime.now()
    print(now.strftime("%d/%m/%Y %H:%M:%S"))
    print('Read image file '+fnameread[j])
    im_nx, im_ny, nlam, pixsize_x, pixsize_y, im, rr_au, phi = read_image_ascii(fnameread[j]) # pixsize, rr_au in cm unit
    # Conversion from erg/s/cm/cm/Hz/ster to Jy/pixel
    #conv  = pixsize_x * pixsize_y / (d_pc*pc)**2. * 1e23
    #im_Jyppix = im.copy()*conv
    #im_Jypbeam=np.zeros_like(im_Jyppix)
    # Set 2D beam
    pixsize_x = pixsize_x/au/d_pc; pixsize_y = pixsize_y/au/d_pc   # Convert pixsize to arcsec unit
    #gaussian_2D_kernel = Gaussian2DKernel(bmaj/pixsize_x/np.sqrt(8*np.log(2)),bmin/pixsize_y/np.sqrt(8*np.log(2)),bpa/180.*np.pi,x_size=151, y_size=151)
    # Convolve the image through the 2D beam
    #for i in range(nlam):
        #im_Jypbeam[i,:,:] = convolve(im_Jyppix[i,:,:].T,gaussian_2D_kernel,boundary='fill',fill_value='0',normalize_kernel=True)
        #gaussian_filter(im_Jyppix[i,:,:],(bmaj*bmin*d_pc/4./2.35)**2*np.pi/4/np.log(2))
    #pixcel_solidangle=(pixsize_x)**2 #pixel size in arcsec
    #beam_solidangle=bmaj*bmin*np.pi/4.0/np.log(2)     # DSHARP 12CO resolution ~ 0.095X0.083 arcsec
    im_Jypbeam=im#_Jypbeam*beam_solidangle/pixcel_solidangle
    im_Jypbeam[np.where(im==-1e91)] = 0.0
    zratio = im_Jypbeam/au/rr_au
    # Make fits header ------------------------------------------------------------------
    # Set velocity axis
    nchans=nlam
    chanstep=0.06
    voff=0.
    chanmin = (-nchans/2.)*chanstep
    velo_steps = np.arange(nchans)*chanstep+chanmin+voff
    now = datetime.now()
    print(now.strftime("%d/%m/%Y %H:%M:%S"))
    print('Writing out '+outfile[j])
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
    hdr['NAXIS1'] = im_nx
    hdr['NAXIS2'] = im_ny
    hdr['NAXIS3'] = nchans
    hdr['NAXIS4'] = 1
    hdr['CDELT1'] = -pixsize_x/3600.
    hdr['CRPIX1'] = im_nx/2.+1
    hdr['CRVAL1'] = hdu.header['CRVAL1']
    hdr['CTYPE1'] = 'RA---SIN'
    hdr['CUNIT1'] = 'deg'
    hdr['CDELT2'] = pixsize_x/3600.
    hdr['CRPIX2'] = im_ny/2.+1
    hdr['CRVAL2'] = hdu.header['CRVAL2']
    hdr['CTYPE2'] = 'DEC--SIN'
    hdr['CUNIT2'] = 'deg'
    hdr['CTYPE3'] = 'VELO-LSR'
    hdr['CDELT3'] = chanstep#*1e3
    hdr['CUNIT3'] = 'km/s'
    hdr['CRPIX3'] = 1.
    hdr['CRVAL3'] = velo_steps[0]#*1e3
    hdr['CTYPE4'] = 'STOKES'
    hdr['CDELT4'] = 1.
    hdr['CRPIX4'] = 1.
    hdr['CRVAL4'] = 1.
    hdr['EPOCH'] = 2000.
    hdr['RESTFREQ'] = freq0[j]*1e9
    hdr['OBJECT'] = 'RU_Lup'
    hdu = fits.PrimaryHDU(zratio,hdr)
    hdu.writeto(outfile[j],overwrite=True,output_verify='fix')
    

