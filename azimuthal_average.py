import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from astropy.io import fits
from astropy import units as u
#import bettermoments.collapse_cube as bm
#from Model_setup_subroutines import *
import argparse
from gofish import imagecube


parser = argparse.ArgumentParser()
parser.add_argument('-file', default='None', type=str, help='Test name. Default is None')
parser.add_argument('-bmaj', default='bmaj5', type=str, help='Beam size of fits. Default is bmaj5')
args = parser.parse_args()
test = args.file
bmaj = args.bmaj

# File name set
fdir = '/Users/kimsj/Documents/RADMC-3D/radmc3d-2.0/RU_Lup_test/Automatics/Fin_script/fiducial_wind/'
mole = ['12CO_2-1','13CO_2-1','13CO_3-2','C18O_2-1','C18O_3-2','CN_3-2','cont220GHz','cont336GHz']
#test = 'fiducial'
#bmaj = 'bmaj51'
# Radius range set
DPA = 121.0; inc = 25.0
r_min = 0.0;  r_max = 250.0   # in au unit
dr = 5.0; nr = int(r_max/dr)
r_bin = np.arange(r_min,r_max+dr,dr)
rc = (r_bin[1:nr+1] + r_bin[0:nr])*0.5
# PA range set
PA_min = -180.0; PA_max = 180.0
# Image center and geometry set
dxc = 0.00; dyc = 0.00
z0 = 0.00;psi = 1.0; z1 = 0.0; phi = 1.0
d_pc = 160.0 # Distant of the source

for i in range(0,len(mole)):
    I_avg = np.zeros_like(rc)
    I_std = np.zeros_like(rc)
    if mole[i] == 'cont220GHz' or mole[i] == 'cont336GHz':
        fitsname = 'RULup_'+mole[i]+'_'+test+'_'+bmaj+'.fits'  #
        #print(fitsname)
    else:
        fitsname = 'RULup_'+mole[i]+'_'+test+'_'+bmaj+'_M0.fits'  #
        #print(fitsname)
    outname = 'RULup_'+mole[i]+'_'+test+'_'+bmaj+'_radial.dat'
    cube = imagecube(fdir + fitsname)
    rvals, tvals, _ = cube.disk_coords(x0=dxc,y0=dyc,inc=inc,PA=DPA,z0=z0,psi=psi,z1=z1,phi=phi)
    r_au = rvals * d_pc
    for j in range(nr):
        r_mask = np.logical_and(r_au >= r_bin[j], r_au <= r_bin[j+1])
        PA_mask = np.logical_and(tvals >= PA_min*np.pi/180., tvals <= PA_max*np.pi/180.)
        #v_mask = np.logical_and(vmodel >= -5.0e3, vmodel <= 5.0e3)
        mask = r_mask*PA_mask#*v_mask
        n_sample = np.sum(mask) ; idx,idy = np.where(mask >0)
        I_sample = np.zeros(n_sample)
        for k in range(len(idx)):
            I_sample[k] = cube.data[idx[k],idy[k]]
        I_avg[j] = I_sample.mean(); I_std[j] = np.std(I_sample)
    outfile=open(fdir + outname, 'w+')
    data=np.array((rc,I_avg,I_std))
    data=data.T
    np.savetxt(outfile, data, fmt='%13.8f', delimiter='  ',newline='\n')
    outfile.close()
    
