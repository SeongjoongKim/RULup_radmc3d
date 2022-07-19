import numpy as np
from Model_setup_subroutines import *
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-file', default='None', type=str, help='input test title')
parser.add_argument('-Tw', default=0.1, type=float, help='input test title')
args = parser.parse_args()

# Read grid file
fnameread      = 'amr_grid.inp'
nr, ntheta, nphi, grids = read_grid_inp(fnameread)
ri = grids[0:nr+1]; thetai = grids[nr+1:nr+ntheta+2]; phii = grids[nr+ntheta+2:]
rc       = 0.5 * ( ri[0:nr] + ri[1:nr+1] )
thetac   = 0.5 * ( thetai[0:ntheta] + thetai[1:ntheta+1] )
phic     = 0.5 * ( phii[0:nphi] + phii[1:nphi+1] )
qq       = np.meshgrid(rc,thetac,phic,indexing='ij')
rr       = qq[0]#.reshape(nr,ntheta)    # cm
tt       = qq[1]#.reshape(nr,ntheta)     # rad
zr       = np.pi/2.e0 - qq[1]     # rad

# Read temperature file
test = args.file
fnameread      = 'Tdust_'+test+'.dat'
im_n, npop, temp = read_T_ascii(fnameread)
temp=temp.reshape((npop,(ntheta)*(nr)))
temp_smlgr=temp[0,:].reshape(nr,ntheta)
temp_biggr=temp[1,:]

mu = 2.34
kb = 1.38e-16
NA = 6.02e23
mH = 1.0/NA
#sigma=np.sqrt(2.*kb*temp_smlgr/mu/mH)
vturb = Tw*np.sqrt(kb*temp_smlgr/mu/mH)  # local soundspeed * Tw

# Write the dust and gas temperature input files
with open('microturbulence.inp','w+') as f:
    f.write('1\n')                       # Format number
    f.write('%d\n'%(nr*ntheta*nphi))     # Nr of cells
    data = vturb.ravel(order='F')        # Create a 1-D view, fortran-style indexing
    data.tofile(f, sep='\n', format="%13.6e")
    f.write('\n')

