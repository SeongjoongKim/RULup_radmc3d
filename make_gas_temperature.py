import numpy as np
from Model_setup_subroutines import *
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-file', default='None', type=str, help='input test title')
args = parser.parse_args()

# Read grid file
fnameread      = 'amr_grid.inp'
nr, ntheta, nphi, grids = read_grid_inp(fnameread)
ri = grids[0:nr+1]; thetai = grids[nr+1:nr+ntheta+2]; phii = grids[nr+ntheta+2:-1]
rc       = 0.5 * ( ri[0:nr] + ri[1:nr+1] )
thetac   = 0.5 * ( thetai[0:ntheta] + thetai[1:ntheta+1] )
phic     = 0.5 * ( phii[0:nphi] + phii[1:nphi+1] )
qq       = np.meshgrid(rc,thetac,phic,indexing='ij')
rr       = qq[0]    # cm
tt       = qq[1]     # rad
zr       = np.pi/2.e0 - qq[1]     # rad

# Read temperature file
test = args.file
fnameread      = 'Tdust_'+test+'.dat'
im_n, npop, temp = read_T_ascii(fnameread)
temp=temp.reshape((npop,(ntheta)*(nr)))
temp_smlgr=temp[0,:]
temp_biggr=temp[1,:]

# Write the dust and gas temperature input files
write_Tdust(temp_biggr,nr,ntheta,nphi)
write_Tgas(temp_smlgr,nr,ntheta,nphi)
