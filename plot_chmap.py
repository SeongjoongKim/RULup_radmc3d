import os,sys
from astropy.io import fits
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Rectangle
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1 import ImageGrid
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredAuxTransformBox
from astropy import units as u
from astropy.coordinates import SkyCoord
import numpy as np
import argparse

# =======================================================================================
# Setup functions
# =======================================================================================
def read_fits(fitsfile):
    Fits_data={}
    hdulist=fits.open(fitsfile)
    prihdr=hdulist[0].header
    image_data=hdulist[0].data
    d=int(prihdr['NAXIS'])
    Dim=[prihdr['NAXIS'+str(i)] for i in range(1,d+1)]
    crval=[prihdr['CRVAL'+str(i)] for i in range(1,d+1)]
    crdel=[prihdr['CDELT'+str(i)] for i in range(1,d+1)]
    bmaj=prihdr['BMAJ'] # degree
    bmin=prihdr['BMIN']
    pa=prihdr['BPA']
    cell=abs(prihdr['CDELT2'])
    Bmaj=(bmaj*u.degree).to(u.arcsec).value # arcsec
    Bmin=(bmin*u.degree).to(u.arcsec).value # arcsec
    Cell=(cell*u.degree).to(u.arcsec).value # arcsec
    x0=(prihdr['CRVAL1']*u.degree).to(u.arcsec).value # center coord in arcsec
    y0=(prihdr['CRVAL2']*u.degree).to(u.arcsec).value
    xlen=prihdr['NAXIS1']*Cell; ylen=prihdr['NAXIS2']*Cell
    extent=(-xlen/2.,xlen/2.,-ylen/2.,ylen/2.)
    xmin=x0-xlen/2.; xmax=x0+xlen/2.
    ymin=y0-ylen/2.; ymax=y0+ylen/2.
    c0 = SkyCoord(ra=x0*u.arcsec, dec=y0*u.arcsec)
    cmin=SkyCoord(ra=xmin*u.arcsec, dec=ymin*u.arcsec)
    cmax=SkyCoord(ra=xmax*u.arcsec, dec=ymax*u.arcsec)
    extent_radec=(cmin.ra.hms.s,cmax.ra.hms.s,cmin.dec.dms.s,cmax.dec.dms.s)
    Fits_data['Dim']=Dim
    Fits_data['Beam']=(Bmaj,Bmin,pa)
    Fits_data['Cell']=Cell
    Fits_data['OBJ']=prihdr['OBJECT']
    Fits_data['NAXIS']=Dim
    Fits_data['CRVAL']=crval
    Fits_data['CDELT']=crdel
    Fits_data['AXRANGE']={'arcsec':extent,'radec':extent_radec}
    try:
        Fits_data['CTYPE3']=(prihdr['ctype3'], prihdr['cunit3'])
    except KeyError:
        Fits_data['CTYPE3']=prihdr['CTYPE3']  #(prihdr['ctype3'], prihdr['cunit3'])
        Fits_data['CRVAL3']=prihdr['CRVAL3']
        Fits_data['CDELT3']=prihdr['CDELT3']  #Fits_data['RANGE3']=(prihdr['CRVAL3'],prihdr['CDELT3'],prihdr['CRVAL3']+prihdr['CDELT3']*prihdr['NAXIS3'])
    try:
        Fits_data['RESTFREQ']=prihdr['RESTFRQ']
    except:
        pass
    try:
        Fits_data['RESTFREQ']=prihdr['RESTFREQ']
    except:
        pass
    Fits_data['DATA']=image_data
    return Fits_data
    

def add_beam(ax,beam,frameon=True):
    (bmaj,bmin,pa)=beam
    box=AnchoredAuxTransformBox(ax.transData, loc=3, frameon=frameon)
    color='k' if frameon else 'w'
    beam_el=Ellipse((0,0),height=bmaj,width=bmin,angle=pa,color=color)
    #beam_el=Ellipse((0,0),height=bmaj,width=bmin,angle=pa,facecolor='w',edgecolor='k',linewidth=0.2)
    box.drawing_area.add_artist(beam_el)
    ax.add_artist(box)


def generate_cmap(colors):
    values = range(len(colors))
    vmax = np.ceil(max(values))
    color_list = []
    for v, c in zip(values, colors):
        color_list.append( ( v/ vmax, c) )
    return LinearSegmentedColormap.from_list('custom_cmap', color_list)


my_cm=generate_cmap(['black','navy','blue','cyan','lime','yellow','red','magenta','white'])

# =======================================================================================
# Read arguments
# =======================================================================================
parser = argparse.ArgumentParser()
parser.add_argument('-mole', default='None', type=str, help='The molecular line name. Default is None.')
parser.add_argument('-bmaj', default='bmaj5', type=str, help='The beam size. Default is bmaj5.')
parser.add_argument('-tname', default='None', type=str, help='Test name. Default is None.')
args = parser.parse_args()
print(args)

mole = args.mole  #'C18O_2-1'
bmaj = args.bmaj  #'bmaj5'
tname = args.tname

# =======================================================================================
# Read files
# =======================================================================================
fdir = '/Users/kimsj/Documents/RADMC-3D/radmc3d-2.0/RU_Lup_test/Automatics/Fin_script/fiducial_wind/'
fitsname = 'RULup_'+mole+'_'+tname+'_'+bmaj+'.fits'
if not os.path.exists(fdir+fitsname):
    fitsname = 'RULup_'+mole+'_fiducial_wind_'+bmaj+'.fits'

save=True               # Whether to save as a figure
savename='RULup_'+mole+'_'+tname+'_'+bmaj+'_chmap.pdf'    # Figure name
if tname == 'fiducial' or tname == 'fiducial_wind':
    savename='RULup_'+mole+'_fiducial_wind_'+bmaj+'_chmap.pdf'    # Figure name

w=5         # Set the number of columns to be drawn
h=6         # Set the number of rows to be drawn
N=w*h       # Total number of the channel to be drawn
start_ch=85 # Set the first channel to be drawn
### for cycle2 start from 5
### for SV, star from another number
lim=2.0      # Set the range of axis (arcsec)
xlim=[-lim,lim]; ylim=xlim
imsize=lim

### Get the cube data ###
fits_cube=read_fits(fdir+fitsname)
image_cube=fits_cube['DATA']
size=fits_cube['NAXIS'][0]
cell=fits_cube['Cell']
edge=cell*size/2
beam=fits_cube['Beam']
x0_cube=(fits_cube['CRVAL'][0]*u.degree).to(u.arcsec).value
y0_cube=(fits_cube['CRVAL'][1]*u.degree).to(u.arcsec).value
Imax=np.nanmax(image_cube[0])
Imin=np.nanmin(image_cube[0])
vmax=Imax; vmin=Imin
#vmax=0.25; vmin=0
col_levels=np.logspace(Imin,Imax,20)

if fits_cube['CTYPE3'] == 'FREQ':
    f0=fits_cube['CRVAL'][2]*1e-9
    df=fits_cube['CDELT'][2]*1e-9
    restfreq=fits_cube['RESTFREQ']*1e-9
    c=299792.458 # speed of light (km/s)
    v0=-(f0-restfreq)/c*restfreq
    dv=-df*c/restfreq
if fits_cube['CTYPE3'] == 'VELO-LSR':
    v0 = fits_cube['CRVAL3']
    dv = fits_cube['CDELT3']
    
extent=fits_cube['AXRANGE']['arcsec']

### Draw the channel map ###
fig=plt.figure(figsize=(4*w,2*h))
grid=ImageGrid(fig,111,nrows_ncols=(h,w),axes_pad=0.,label_mode='1', share_all=True, cbar_location='right', cbar_mode='single',cbar_size='3%',cbar_pad='1%')
for i,ax in enumerate(grid):
    if len(image_cube.shape) == 4:
        im=ax.imshow(image_cube[0][start_ch+i],extent=extent,origin='lower',vmax=vmax,vmin=vmin,cmap="gist_heat")
    if len(image_cube.shape) == 3:
        im=ax.imshow(image_cube[start_ch+i],extent=extent,origin='lower',vmax=vmax,vmin=vmin,cmap="gist_heat")
    v=v0+dv*(start_ch+i)
    txt='%.2f km/s' % v
    ax.text(0.95, 0.95, txt, ha='right', va='top', transform=ax.transAxes, color="w")
    ax.set_xlim(xlim); ax.set_ylim(ylim)
    if i==w*(h-1):
        ax.set_xlabel(r"$\Delta\alpha\;(\mathrm{arcsec})$")
        ax.set_ylabel(r"$\Delta\delta\;(\mathrm{arcsec})$")
        add_beam(ax,beam,frameon=False)
grid.cbar_axes[0].colorbar(im)
grid.cbar_axes[0].set_ylabel(r'$\mathrm{Jy\ beam^{-1}}$')
#grid.cbar_axes[0].set_yticks((vmin,vmax))

if save:
    plt.savefig('./chmaps/'+savename, bbox_inches='tight', pad_inches=0.1)
else:
    plt.show()
