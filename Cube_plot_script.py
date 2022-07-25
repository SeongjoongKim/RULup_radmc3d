from Kim_subroutines import *
import argparse
import emcee

# =======================================================================================
# Parameter setup
# =======================================================================================
parser = argparse.ArgumentParser()
parser.add_argument('-mole', default='None', type=str, help='The molecular line name. Default is None.')
parser.add_argument('-ftype', default='mod', type=str, help='The file type. Default is mod.')
parser.add_argument('-wc', default='F', type=str, help='If you want to include continuum, set this argument T. default is F (no continuum).')
args = parser.parse_args()
#mole =  args.mole  
#ftype =  args.ftype  
mole1 = 'CN_3-2'
mole2 = 'CN_3-2'
ftype1 = 'mod'
ftype2 = 'mod'
test1 = 'nowind_turb5'
test2 = 'nowind'
bmaj = 'bmaj5'
# Radial range for extracting the spectrum
r_in = 0.05; r_out = 0.15; PA_width = 10
func1 = Double; func2 = Double

# =======================================================================================
# Disk parameters
# =======================================================================================
mstar, rstar, tstar = [ 0.63*ms, 2.42*rs, 4073. ] #Stellar parameters. The list has [M_star, R_star, T_star] in Solar mass, Solar radius, and Kelvin unit,
inc = 25.0; DPA = 121.0
PA_min = -180.0; PA_max = 180.0
dxc = 0.00; dyc = 0.00
z0 = 0.0; psi = 1.25; z1 = -0.0; phi = 1.25
r_taper = np.inf; q_taper = 1.0
ring1 = 120.*2.; ring2 = 260.*2.
d_pc = 160.0 # Distant of the source in pc
vsys = 4.44   # System velocity in km/s
print(mole1, ftype1, mole2, ftype2, test1, test2, bmaj, z0, psi)

# Output directory of Gaussian fitting parameter summary
outdir = './teardrop/spec/z0{:3.2f}_r{:3.2f}_{:3.2f}_averaged_spec_{:03}width/'.format(z0,r_in,r_out,PA_width)
if not os.path.exists(outdir):
    os.mkdir(outdir)
    print("Directory " , outdir ,  " Created ")
else:
    print("Directory " , outdir ,  " already exists")

# =======================================================================================
# Read observational data
# =======================================================================================
if ftype1 == 'obs':
    # Observation data
    fdir = '/Users/kimsj/Documents/RU_Lup/Fin_fits/'
    fitsname = mole1+'_selfcal_matched_cube500.fits'  
if ftype1 == 'mod':
    # Model data
    fdir = '/Users/kimsj/Documents/RADMC-3D/radmc3d-2.0/RU_Lup_test/Automatics/Fin_script/fiducial_wind/'
    if test1 =='Cw1e-5':
        fitsname = 'RULup_'+mole1+'_fiducial_wind_sigma3_'+bmaj+'.fits'
    else:
        fitsname = 'RULup_'+mole1+'_fiducial_wind_sigma3_'+test1+'_'+bmaj+'.fits'

cube = datacube(fdir+fitsname)

# Apply continuum subtraction? # =================================================================
nchan_c = 10; start_c = 5
if args.wc == 'F':
    data_cl = np.zeros((cube.ny,cube.nx))
    data_cr = np.zeros((cube.ny,cube.nx))
    for i in range(nchan_c):
        data_cl += cube.data[start_c+i,:,:]
        data_cr += cube.data[cube.nv-1-start_c-i,:,:]
    data_cl /= nchan_c; data_cr /= nchan_c
    data_c = (data_cl+data_cr)*0.5
    cube.data -= data_c[None,:,:]  # continuum subtraction

# =======================================================================================
# Plot channel map
# =======================================================================================
#w = 1; h = 1
#nchan = int(cube.nv/2.)-int(w*h/2.)
#cube.plot_chmap(w, h, nchan=nchan, lim=1.5, figname='./teardrop/'+mole1+'_chmap_mask.pdf', save=True)

# =======================================================================================
# Make moment maps
# =======================================================================================
# RMS levels
# 12CO 2-1: 3.0e-3   ;  13CO 2-1: 3.0e-3  ; 13CO 3-2: 6.5e-3  ;  C18O 2-1: 2/2e-3  ; C18O 3-2: 6.0e-3  ; CN 3-2: 5.5e-3
rms = 1.0e-5; i_start = 0; i_end = cube.nv; N_cut = 1
M0, dM0 = cube.collapse_zeroth(rms, i_start, i_end)
M1, dM1 = cube.collapse_first(rms, N_cut, i_start, i_end)
#M2, dM2 = cube.collapse_second(rms, N_cut, i_start, i_end)
#M8, dM8 = cube.collapse_eighth(rms)
'''
# =======================================================================================
# plot moment maps
# =======================================================================================
cube.plot_2Dmap(M0, inc, DPA, d_pc, 1e-3*M0.max(), M0.max(), scale='log', cmap='gist_heat', unit='M0', 
                R1=ring1, RC1='k', R2=ring2, RC2='cyan', figname='None', save=False)
if ftype1 == 'obs': vel_min = 3.5; vel_max = 6.5
if ftype1 == 'mod': vel_min = -1.5; vel_max = 1.5
cube.plot_2Dmap(M1, inc, DPA, d_pc, vel_min, vel_max, scale='linear', cmap='jet', unit='V', 
                R1=ring1, RC1='k', R2=ring2, RC2='gray', figname='None', save=False)
cube.plot_2Dmap(M2, inc, DPA, d_pc, 1e-2*M2.max(), M2.max(), scale='log', cmap='gist_heat', unit='M2', 
                R1=ring1, RC1='cyan', R2=ring2, RC2='k', figname='None', save=False)
'''
# =======================================================================================
# vector projection methods
# =======================================================================================
# Axes rotation from disk plane to sky plane
# inclination: +z moves to +y axis
# DPA: +x moves to +y axis
# The disk inclined and rotated inversely. +y axis becomes near side & disk major axis rotates PA toward -y axis.
ux_sky, uy_sky, uz_sky = rotated_axis(inc, DPA)
print(ux_sky, uy_sky, uz_sky)
# Axes rotation from sky plane to disk plane
ux_disk, uy_disk, uz_disk = np.linalg.inv(np.array((ux_sky,uy_sky,uz_sky)))
print(ux_disk, uy_disk, uz_disk)

# =======================================================================================
# Get coordinates in sky & disk plane
# =======================================================================================
x_sky, y_sky = cube.get_sky_coord()
xx_disk, yy_disk, zz_disk = cube.get_disk_coord(inc, DPA, z0=z0, psi=psi, z1=z1, phi=phi, rt0=r_taper, qt0=q_taper)
#rr_disk = np.hypot(xx_disk,yy_disk)

# =======================================================================================
# Get Velocity fields projected on the sky plane
# =======================================================================================
vkep_sky = cube.get_vkep(inc, DPA, mstar=mstar, d_pc=160.0, z0=z0, psi=psi, z1=z1, phi=phi, rt0=r_taper, qt0=q_taper)
vmod = Gauss_convol(vkep_sky, cube.bmaj, cube.bmin, cube.bpa, cube.pixsize_x, cube.pixsize_y, x_size=151, y_size=151)
#vmod = vkep_sky
vmod *=1e-5
'''
# =======================================================================================
# Plotting the rotation field
# =======================================================================================
figname = './teardrop/'+mole1+'_Flared_emitting_layer_{:3.2f}_{:3.2f}_vkep_vector.pdf'.format(z0,psi)
# Ellipse angle moves toward +y axis from +x axis
wcs = WCS(cube.header).slice([cube.nx,cube.ny])                         # calculate RA, Dec
disk = Ellipse( (int(cube.nx/2), int(cube.ny/2)), ring1/d_pc*100, ring1/d_pc*100*np.cos(inc*np.pi/180.), angle=180.-DPA, edgecolor='white', facecolor='none',ls='--')  # Keplerian disk
ring = Ellipse( (int(cube.nx/2), int(cube.ny/2)), ring2/d_pc*100, ring2/d_pc*100*np.cos(inc*np.pi/180.), angle=180.-DPA, edgecolor='black', facecolor='none',ls='--')   # Outer envelope boundary
fig=plt.figure(num=None, figsize=(8,6), dpi=100, facecolor='w', edgecolor='k')
ax = fig.add_subplot(projection=wcs)
im=ax.imshow(vmod.T,origin='lower',cmap="jet",vmax=1,vmin=-1)
TT = ax.contour(vmod.T,[-0.75,-0.5,-0.3,-0.1,0.0,0.1,0.3,0.5,0.75],linestyles='solid',colors='gray')
plt.clabel(TT)
ax.set_xlabel('RA',fontsize=15)
ax.set_ylabel('Dec',fontsize=15)
ax.add_patch(disk)
ax.add_patch(ring)
#ax.text(0.95, 0.95, freq[i], ha='right', va='top', transform=ax.transAxes, color="w",fontsize=15)
ax.tick_params(axis='both',which='both',length=4,width=1,labelsize=12,direction='in',color='w')
#ax.margins(x=-0.375,y=-0.375)
cbar=plt.colorbar(im, shrink=0.9)
cbar.set_label(r'$\mathrm{(km\ s^{-1})}$', size=10)
plt.savefig(figname, bbox_inches='tight', pad_inches=0.1,dpi=100)
#plt.show()
plt.clf()
'''
#vwind_sky = cube.get_vwind()

# =======================================================================================
# Testing averaged spectra around the minor axis
# =======================================================================================
vref = (np.arange(cube.nv+1)-0.5)*cube.dv + cube.v0
vcent = np.average([vref[1:], vref[:-1]], axis=0)

# Spectra all =============================================================
aver_spec_all, std_spec_all = cube.get_azi_aver_spec(vref, inc, DPA, vmod, r_in, r_out, PA_min=-180.0, PA_max=180.0, 
                      z0=z0, psi=psi, z1=z1, phi=phi, rt0=r_taper, qt0=q_taper)
# Spectra around 0 degree =============================================================
aver_spec_0, std_spec_0 = cube.get_azi_aver_spec(vref, inc, DPA, vmod, r_in, r_out, PA_min=-5.0, PA_max=5.0, 
                      z0=z0, psi=psi, z1=z1, phi=phi, rt0=r_taper, qt0=q_taper)
# Spectra around 180 degree =============================================================
aver_spec_180, std_spec_180 = cube.get_azi_aver_spec(vref, inc, DPA, vmod, r_in, r_out, PA_min=-180.0, PA_max=-175.0, 
                      z0=z0, psi=psi, z1=z1, phi=phi, rt0=r_taper, qt0=q_taper)
# Spectra around -90 degree =============================================================
PA_cen = -90.0; PA_width = 5.0
aver_spec_m90, std_spec_m90 = cube.get_azi_aver_spec(vref, inc, DPA, vmod, r_in, r_out, PA_min=PA_cen - PA_width, PA_max=PA_cen + PA_width, 
                      z0=z0, psi=psi, z1=z1, phi=phi, rt0=r_taper, qt0=q_taper)
# Spectra around +90 degree ===============================================================
PA_cen = 90.0; PA_width = 5.0
aver_spec_p90, std_spec_p90 = cube.get_azi_aver_spec(vref, inc, DPA, vmod, r_in, r_out, PA_min=PA_cen - PA_width, PA_max=PA_cen + PA_width, 
                      z0=z0, psi=psi, z1=z1, phi=phi, rt0=r_taper, qt0=q_taper)

# Gaussian fitting of spectra ===============================================================
# Set the functional form & initial contidions, lower and upper limit of params
if func1 == Double:
    LL = [0.0, -10.0 ,0.0, 0.0, -10.0, 0.0, -1.0]
    UL = [1.0, 10.0, 10.0, 1.0, 10.0, 10.0, 1.0]
    if ftype1 == 'obs': IC = [0.1, 4.5, 0.25, 0.04, 4.0, 0.25, 0.0]
    if ftype1 == 'mod': IC = [0.1, 0.0, 0.25, 0.04, -0.7, 0.25, 0.0]
if func1 == Gaussian:
    LL = [0.0, -10.0 ,0.0, -1.0]
    UL = [1.0, 10.0, 10.0, 1.0]
    if ftype2 == 'obs': IC = [0.1, 4.5, 0.25, 0.04]
    if ftype2 == 'mod': IC = [0.1, 0.0, 0.25, 0.04]

popt0, pcov0 = Fit_spectra(vcent, aver_spec_0, func1, IC, LL, UL)
#IC = [0.1, 4.5, 1.0, -0.01, 5.5, 0.5, 0.0]
popt180, pcov180 = Fit_spectra(vcent, aver_spec_180, func1, IC, LL, UL)
#IC = [0.1, 4.5, 1.0, -0.01, 5.5, 0.5, 0.0]
poptm90, pcovm90 = Fit_spectra(vcent, aver_spec_m90, func1, IC, LL, UL)
#IC = [0.1, 4.5, 1.0, 0.01, 5.5, 0.5, 0.0]
poptp90, pcovp90 = Fit_spectra(vcent, aver_spec_p90, func1, IC, LL, UL)
print( np.array((popt0, popt180, poptm90, poptp90)) )

# MCMC fitting ===============================================================


# Plot the averaged spectra ===============================================================
plt.figure(figsize=(12,8))
plt.subplot(231)
plt.plot(vcent,aver_spec_all,'k', label='All')
plt.legend(prop={'size':7},loc=0)
plt.subplot(232)
plt.plot(vcent,aver_spec_0,'g', label='0')
plt.plot(vcent,func1(vcent,*popt0),'g--', label='Fit')
if func1 == Double: plt.plot(vcent,Gaussian(vcent,popt0[0],popt0[1],popt0[2],popt0[6]),'g--')
if func1 == Double: plt.plot(vcent,Gaussian(vcent,popt0[3],popt0[4],popt0[5],popt0[6]),'g--')
plt.legend(prop={'size':7},loc=0)
plt.subplot(233)
plt.plot(vcent,aver_spec_180,'m', label='180')
plt.plot(vcent,func1(vcent,*popt180),'m--')
if func1 == Double: plt.plot(vcent,Gaussian(vcent,popt180[0],popt180[1],popt180[2],popt180[6]),'m--')
if func1 == Double: plt.plot(vcent,Gaussian(vcent,popt180[3],popt180[4],popt180[5],popt180[6]),'m--')
plt.legend(prop={'size':7},loc=0)
plt.subplot(234)
plt.plot(vcent,aver_spec_m90,'r', label='-90')
plt.plot(vcent,func1(vcent,*poptm90),'r--')
if func1 == Double: plt.plot(vcent,Gaussian(vcent,poptm90[0],poptm90[1],poptm90[2],poptm90[6]),'r--')
if func1 == Double: plt.plot(vcent,Gaussian(vcent,poptm90[3],poptm90[4],poptm90[5],poptm90[6]),'r--')
plt.legend(prop={'size':7},loc=0)
plt.subplot(235)
plt.plot(vcent,aver_spec_p90,'b', label='+90')
plt.plot(vcent,func1(vcent,*poptp90),'b--')
if func1 == Double: plt.plot(vcent,Gaussian(vcent,poptp90[0],poptp90[1],poptp90[2],poptp90[6]),'b--')
if func1 == Double: plt.plot(vcent,Gaussian(vcent,poptp90[3],poptp90[4],poptp90[5],poptp90[6]),'b--')
plt.legend(prop={'size':7},loc=0)
#plt.show()
plt.savefig(outdir+mole1+'_averaged_spec_'+test1+'_'+ftype1+'_'+bmaj+'.pdf',dpi=100,bbox_inches='tight')
plt.clf()

plt.plot(vcent,aver_spec_m90 - aver_spec_p90,'k--',label='-90 - +90')
plt.plot(vcent,aver_spec_0 - aver_spec_180,'g--',label='0 - 180')
plt.plot(vcent,aver_spec_m90 - aver_spec_all,'r--',label='-90 - all')
plt.plot(vcent,aver_spec_p90 - aver_spec_all,'b--',label='+90 - all')
plt.legend(prop={'size':7},loc=0)
#plt.show()
plt.savefig(outdir+mole1+'_spec_subtract_'+test1+'_'+ftype1+'_'+bmaj+'.pdf',dpi=100,bbox_inches='tight')
plt.clf()
'''
# =======================================================================================
# Checking the mask region  
# =======================================================================================
r, theta = cube.get_disk_rtheta(inc, DPA, z0=z0, psi=psi, z1=z1, phi=phi, rt0=r_taper, qt0=q_taper)
r = r.T; theta = theta.T
mask1 = cube.get_mask(r, theta, r_in, r_out, -95, -85)
mask2 = cube.get_mask(r, theta, r_in, r_out, 85, 95)
mask3 = cube.get_mask(r, theta, r_in, r_out, -5.0, 5.0)
mask4 = cube.get_mask(r, theta, r_in, r_out, -180.0, -175.0)
#mask = mask1 + mask2 + mask3 + mask4
wcs = WCS(cube.header).slice([cube.nx,cube.ny])                         # calculate RA, Dec
fig=plt.figure(num=None, figsize=(6,4), dpi=150, facecolor='w', edgecolor='k')
ax = fig.add_subplot(projection=wcs)
#im=ax.imshow(M0,origin='lower',cmap="gist_heat",vmax=M0.max(),vmin=1e-2*M0.max())
#im=ax.imshow(vmod,origin='lower',cmap="jet",vmax=1.0,vmin=-1.0)
#im=ax.imshow(M1,origin='lower',cmap="jet",vmax=1.0,vmin=-1.0)
im=ax.imshow(cube.data[int(cube.nv/2),:,:],origin='lower',cmap="gist_heat",vmax=cube.data.max(),vmin=1e-2*cube.data.max())
ax.scatter(250,250,marker='o',s=10)
TT = ax.contour(mask1,[1],linestyles='solid',colors='b')
TT = ax.contour(mask2,[1],linestyles='solid',colors='r')
TT = ax.contour(mask3,[1],linestyles='solid',colors='g')
TT = ax.contour(mask4,[1],linestyles='solid',colors='m')
ax.set_xlabel('RA',fontsize=15)
ax.set_ylabel('Dec',fontsize=15)
ax.tick_params(axis='both',which='both',length=4,width=1,labelsize=12,direction='in',color='w')
ax.margins(x=-0.25,y=-0.25)
cbar=plt.colorbar(im, shrink=0.9)
cbar.set_label(r'$\mathrm{(Jy\ beam^{-1})}$', size=10)
plt.savefig('./teardrop/Spec/'+mole1+'_z0{:3.2f}_masking_'.format(z0)+test1+'_'+ftype1+'_'+bmaj+'.pdf',dpi=100,bbox_inches='tight')
plt.clf()
'''
# =======================================================================================
# Testing averaged spectra around the minor axis of comparing models
# =======================================================================================
if ftype2 == 'obs':
    # Observation data
    fdir = '/Users/kimsj/Documents/RU_Lup/Fin_fits/'
    fitsname = mole2+'_selfcal_matched_cube500.fits'
if ftype2 == 'mod':
    # Model data
    fdir = '/Users/kimsj/Documents/RADMC-3D/radmc3d-2.0/RU_Lup_test/Automatics/Fin_script/fiducial_wind/'
    if test2 =='Cw1e-5':
        fitsname = 'RULup_'+mole2+'_fiducial_wind_sigma3_'+bmaj+'.fits'
    else:
        fitsname = 'RULup_'+mole2+'_fiducial_wind_sigma3_'+test2+'_'+bmaj+'.fits'

cube2 = datacube(fdir+fitsname)

nchan_c = 10; start_c = 5
if args.wc == 'F':
    data_cl = np.zeros((cube2.ny,cube2.nx))
    data_cr = np.zeros((cube2.ny,cube2.nx))
    for i in range(nchan_c):
        data_cl += cube2.data[start_c+i,:,:]
        data_cr += cube2.data[cube2.nv-1-start_c-i,:,:]
    data_cl /= nchan_c; data_cr /= nchan_c
    data_c = (data_cl+data_cr)*0.5
    cube2.data -= data_c[None,:,:]  # continuum subtraction

# Set vel axis # ========================================================================
vref2 = (np.arange(cube2.nv+1)-0.5)*cube2.dv + cube2.v0
vcent2 = np.average([vref2[1:], vref2[:-1]], axis=0)

# Spectra all =============================================================
aver_spec_all2, std_spec_all2 = cube2.get_azi_aver_spec(vref2, inc, DPA, vmod, r_in, r_out, PA_min=-180.0, PA_max=180.0, 
                      z0=z0, psi=psi, z1=z1, phi=phi, rt0=r_taper, qt0=q_taper)
# Spectra around 0 dgeree =============================================================
aver_spec_02, std_spec_02 = cube2.get_azi_aver_spec(vref2, inc, DPA, vmod, r_in, r_out, PA_min=-5.0, PA_max=5.0, 
                      z0=z0, psi=psi, z1=z1, phi=phi, rt0=r_taper, qt0=q_taper)
# Spectra around 180 degree =============================================================
aver_spec_1802, std_spec_1802 = cube2.get_azi_aver_spec(vref2, inc, DPA, vmod, r_in, r_out, PA_min=-180.0, PA_max=-175.0, 
                      z0=z0, psi=psi, z1=z1, phi=phi, rt0=r_taper, qt0=q_taper)
# Spectra around -90 degree =============================================================
PA_cen = -90.0; PA_width = 5.0
aver_spec_m902, std_spec_m902 = cube2.get_azi_aver_spec(vref2, inc, DPA, vmod, r_in, r_out, PA_min=PA_cen - PA_width, PA_max=PA_cen + PA_width, 
                      z0=z0, psi=psi, z1=z1, phi=phi, rt0=r_taper, qt0=q_taper)
# Spectra around +90 degree ===============================================================
PA_cen = 90.0; PA_width = 5.0
aver_spec_p902, std_spec_p902 = cube2.get_azi_aver_spec(vref2, inc, DPA, vmod, r_in, r_out, PA_min=PA_cen - PA_width, PA_max=PA_cen + PA_width, 
                      z0=z0, psi=psi, z1=z1, phi=phi, rt0=r_taper, qt0=q_taper)

if ftype1 == 'obs':
    if ftype2 =='mod': vcent2 += vsys
    aver_spec_all2 = interp1d(vcent2, aver_spec_all2, bounds_error=False)(vcent)
    aver_spec_all2[np.isnan(aver_spec_all2)] = 0.0
    aver_spec_02 = interp1d(vcent2, aver_spec_02, bounds_error=False)(vcent)
    aver_spec_02[np.isnan(aver_spec_02)] = 0.0
    aver_spec_1802 = interp1d(vcent2, aver_spec_1802, bounds_error=False)(vcent)
    aver_spec_1802[np.isnan(aver_spec_1802)] = 0.0
    aver_spec_m902 = interp1d(vcent2, aver_spec_m902, bounds_error=False)(vcent)
    aver_spec_m902[np.isnan(aver_spec_m902)] = 0.0
    aver_spec_p902 = interp1d(vcent2, aver_spec_p902, bounds_error=False)(vcent)
    aver_spec_p902[np.isnan(aver_spec_p902)] = 0.0
    vcent2 = vcent

# Gaussian fitting of spectra ===============================================================
# Set the functional form & initial contidions, lower and upper limit of params
if func2 == Double:
    LL = [0.0, -10.0 ,0.0, 0.0, -10.0, 0.0, -1.0]
    UL = [1.0, 10.0, 10.0, 1.0, 10.0, 10.0, 1.0]
    if ftype1 == 'obs': IC = [0.1, 4.5, 0.25, 0.04, 4.0, 0.25, 0.0]
    if ftype1 == 'mod': IC = [0.1, 0.0, 0.25, 0.04, -0.7, 0.25, 0.0]
if func2 == Gaussian:
    LL = [0.0, -10.0 ,0.0, -1.0]
    UL = [1.0, 10.0, 10.0, 1.0]
    if ftype2 == 'obs': IC = [0.1, 4.5, 0.25, 0.04]
    if ftype2 == 'mod': IC = [0.1, 0.0, 0.25, 0.04]

popt0, pcov0 = Fit_spectra(vcent2, aver_spec_02, func2, IC, LL, UL)
#IC = [0.1, 0.0, 1.0, 0.0]
popt180, pcov180 = Fit_spectra(vcent2, aver_spec_1802, func2, IC, LL, UL)
#IC = [0.1, -0.5, 1.0, 0.0]
poptm90, pcovm90 = Fit_spectra(vcent2, aver_spec_m902, func2, IC, LL, UL)
#IC = [0.1, 0.5, 1.0, 0.0]
poptp90, pcovp90 = Fit_spectra(vcent2, aver_spec_p902, func2, IC, LL, UL)
print(np.array((popt0, popt180, poptm90, poptp90)) )

plt.figure(figsize=(12,8))
plt.subplot(231)
plt.plot(vcent2,aver_spec_all2,'k', label='All')
plt.legend(prop={'size':7},loc=0)
plt.subplot(232)
plt.plot(vcent2,aver_spec_02,'g', label='0')
plt.plot(vcent2, func2(vcent2, *popt0),'g--')
if func2 == Double: plt.plot(vcent2,Gaussian(vcent2,popt0[0],popt0[1],popt0[2],popt0[6]),'g--')
if func2 == Double: plt.plot(vcent2,Gaussian(vcent2,popt0[3],popt0[4],popt0[5],popt0[6]),'g--')
plt.legend(prop={'size':7},loc=0)
plt.subplot(233)
plt.plot(vcent2,aver_spec_1802,'m', label='180')
plt.plot(vcent2, func2(vcent2, *popt180),'m--')
if func2 == Double: plt.plot(vcent2,Gaussian(vcent2,popt180[0],popt180[1],popt180[2],popt180[6]),'m--')
if func2 == Double: plt.plot(vcent2,Gaussian(vcent2,popt180[3],popt180[4],popt180[5],popt180[6]),'m--')
plt.legend(prop={'size':7},loc=0)
plt.subplot(234)
plt.plot(vcent2,aver_spec_m902,'r', label='-90')
plt.plot(vcent2, func2(vcent2, *poptm90),'r--')
if func2 == Double: plt.plot(vcent2,Gaussian(vcent2,poptm90[0],poptm90[1],poptm90[2],poptm90[6]),'r--')
if func2 == Double: plt.plot(vcent2,Gaussian(vcent2,poptm90[3],poptm90[4],poptm90[5],poptm90[6]),'r--')
plt.legend(prop={'size':7},loc=0)
plt.subplot(235)
plt.plot(vcent2,aver_spec_p902,'b', label='+90')
plt.plot(vcent2, func2(vcent2, *poptp90),'b--')
if func2 == Double: plt.plot(vcent2,Gaussian(vcent2,poptp90[0],poptp90[1],poptp90[2],poptp90[6]),'b--')
if func2 == Double: plt.plot(vcent2,Gaussian(vcent2,poptp90[3],poptp90[4],poptp90[5],poptp90[6]),'b--')
plt.legend(prop={'size':7},loc=0)
#plt.show()
plt.savefig(outdir+mole2+'_averaged_spec_'+test2+'_'+ftype2+'_'+bmaj+'.pdf',dpi=100,bbox_inches='tight')
plt.clf()

plt.plot(vcent2,aver_spec_m902 - aver_spec_p902,'k--',label='-90 - +90')
plt.plot(vcent2,aver_spec_02 - aver_spec_1802,'g--',label='0 - 180')
plt.plot(vcent2,aver_spec_m902 - aver_spec_all2,'r--',label='-90 - all')
plt.plot(vcent2,aver_spec_p902 - aver_spec_all2,'b--',label='+90 - all')
plt.legend(prop={'size':7},loc=0)
#plt.show()
plt.savefig(outdir+mole2+'_spec_subtract_'+test2+'_'+ftype2+'_'+bmaj+'.pdf',dpi=100,bbox_inches='tight')
plt.clf()

# ================================================================================================
# Comparison between the different molecular lines with the same model parameter
if mole1 != mole2 and test1 == test2:
    plt.figure(figsize=(12,8))
    plt.subplot(231)
    plt.plot(vcent2,aver_spec_all - aver_spec_all2,'k--',label='all')
    plt.legend(prop={'size':7},loc=0)
    plt.subplot(232)
    plt.plot(vcent,aver_spec_0,'g', label=mole1)
    plt.plot(vcent2,aver_spec_02,'g:', label=mole2)
    plt.plot(vcent2,aver_spec_0 - aver_spec_02,'g--',label='diff,0')
    plt.legend(prop={'size':7},loc=0)
    plt.subplot(233)
    plt.plot(vcent,aver_spec_180,'m', label=mole1)
    plt.plot(vcent2,aver_spec_1802,'m:', label=mole2)
    plt.plot(vcent2,aver_spec_180 - aver_spec_1802,'m--',label='diff,180')
    plt.legend(prop={'size':7},loc=0)
    plt.subplot(234)
    plt.plot(vcent,aver_spec_m90,'r', label=mole1)
    plt.plot(vcent2,aver_spec_m902,'r:', label=mole2)
    plt.plot(vcent2,aver_spec_m90 - aver_spec_m902,'r--',label='diff,-90')
    plt.legend(prop={'size':7},loc=0)
    plt.subplot(235)
    plt.plot(vcent,aver_spec_p90,'b', label=mole1)
    plt.plot(vcent2,aver_spec_p902,'b:', label=mole2)
    plt.plot(vcent2,aver_spec_p90 - aver_spec_p902,'b--',label='diff,+90')
    plt.legend(prop={'size':7},loc=0)
    #plt.show()
    plt.savefig(outdir+mole1+'_'+ftype1+'-'+mole2+'_'+ftype2+'_spec_subtract_'+test1+'_'+bmaj+'.pdf',dpi=100,bbox_inches='tight')
    plt.clf()

# Comparison between the normalized molecular lines with the same model parameter
if mole1 != mole2 and test1 == test2:
    plt.figure(figsize=(12,8))
    plt.subplot(231)
    plt.plot(vcent2,aver_spec_all - aver_spec_all2,'k--',label='all')
    plt.legend(prop={'size':7},loc=0)
    plt.subplot(232)
    plt.plot(vcent,aver_spec_0/aver_spec_0.max(),'g', label=mole1)
    plt.plot(vcent2,aver_spec_02/aver_spec_02.max(),'g:', label=mole2)
    plt.plot(vcent2,aver_spec_0/aver_spec_0.max() - aver_spec_02/aver_spec_02.max(),'g--',label='diff,0')
    plt.legend(prop={'size':7},loc=0)
    plt.subplot(233)
    plt.plot(vcent,aver_spec_180/aver_spec_180.max(),'m', label=mole1)
    plt.plot(vcent2,aver_spec_1802/aver_spec_1802.max(),'m:', label=mole2)
    plt.plot(vcent2,aver_spec_180/aver_spec_180.max() - aver_spec_1802/aver_spec_1802.max(),'m--',label='diff,180')
    plt.legend(prop={'size':7},loc=0)
    plt.subplot(234)
    plt.plot(vcent,aver_spec_m90/aver_spec_m90.max(),'r', label=mole1)
    plt.plot(vcent2,aver_spec_m902/aver_spec_m902.max(),'r:', label=mole2)
    plt.plot(vcent2,aver_spec_m90/aver_spec_m90.max() - aver_spec_m902/aver_spec_m902.max(),'r--',label='diff,-90')
    plt.legend(prop={'size':7},loc=0)
    plt.subplot(235)
    plt.plot(vcent,aver_spec_p90/aver_spec_p90.max(),'b', label=mole1)
    plt.plot(vcent2,aver_spec_p902/aver_spec_p902.max(),'b:', label=mole2)
    plt.plot(vcent2,aver_spec_p90/aver_spec_p90.max() - aver_spec_p902/aver_spec_p902.max(),'b--',label='diff,+90')
    plt.legend(prop={'size':7},loc=0)
    #plt.show()
    plt.savefig(outdir+mole1+'_'+ftype1+'-'+mole2+'_'+ftype2+'_normal_spec_subtract_'+test1+'_'+bmaj+'.pdf',dpi=100,bbox_inches='tight')
    plt.clf()

# ================================================================================================
# Comparison the same molecular line between two models
if mole1 == mole2 and test1 != test2:
    # Comapring the observation vs model
    #if ftype == 'obs': vcent2 += vsys
    # Ploting the original spectra
    plt.figure(figsize=(10,8))
    plt.subplot(231)
    plt.plot(vcent2,aver_spec_all2,'k--', label='all,'+test2)
    if ftype1 == 'obs':
        plt.plot(vcent,aver_spec_all,'k', label='all,obs')
    else:
        plt.plot(vcent,aver_spec_all,'k', label='all,'+test1)
    plt.legend(prop={'size':7},loc=0)
    plt.subplot(232)
    plt.plot(vcent2,aver_spec_02,'g--')
    plt.plot(vcent,aver_spec_0,'g', label='0')
    plt.plot(vcent,aver_spec_0-aver_spec_02,'g:')
    plt.legend(prop={'size':7},loc=0)
    plt.subplot(233)
    plt.plot(vcent2,aver_spec_1802,'m--')
    plt.plot(vcent,aver_spec_180,'m', label='180')
    plt.plot(vcent,aver_spec_180-aver_spec_1802,'m:')
    plt.legend(prop={'size':7},loc=0)
    plt.subplot(234)
    plt.plot(vcent2,aver_spec_m902,'r--')
    plt.plot(vcent,aver_spec_m90,'r', label='-90')
    plt.plot(vcent,aver_spec_m90-aver_spec_m902,'r:')
    plt.legend(prop={'size':7},loc=0)
    plt.subplot(235)
    plt.plot(vcent2,aver_spec_p902,'b--')
    plt.plot(vcent,aver_spec_p90,'b', label='+90')
    plt.plot(vcent,aver_spec_p90-aver_spec_p902,'b:')
    plt.legend(prop={'size':7},loc=0)
    plt.savefig(outdir+mole1+'_'+test1+'_'+ftype1+'-'+test2+'_'+ftype2+'_spec_comparison_'+bmaj+'.pdf',dpi=100,bbox_inches='tight')
    plt.clf()
    
# Comparison the normalized same molecular line between two models
if mole1 == mole2 and test1 != test2:
    plt.figure(figsize=(10,8))
    plt.subplot(231)
    plt.plot(vcent2,aver_spec_all2/aver_spec_all2.max(),'k--', label='all,'+test2)
    if ftype1 == 'obs':
        plt.plot(vcent,aver_spec_all/aver_spec_all.max(),'k', label='all,obs')
    else:
        plt.plot(vcent,aver_spec_all/aver_spec_all.max(),'k', label='all,'+test1)
    plt.legend(prop={'size':7},loc=0)
    plt.subplot(232)
    plt.plot(vcent2,aver_spec_02/aver_spec_02.max(),'g--')
    plt.plot(vcent,aver_spec_0/aver_spec_0.max(),'g', label='0')
    plt.plot(vcent,aver_spec_0/aver_spec_0.max()-aver_spec_02/aver_spec_02.max(),'g:')
    plt.legend(prop={'size':7},loc=0)
    plt.subplot(233)
    plt.plot(vcent2,aver_spec_1802/aver_spec_1802.max(),'m--')
    plt.plot(vcent,aver_spec_180/aver_spec_180.max(),'m', label='180')
    plt.plot(vcent,aver_spec_180/aver_spec_180.max()-aver_spec_1802/aver_spec_1802.max(),'m:')
    plt.legend(prop={'size':7},loc=0)
    plt.subplot(234)
    plt.plot(vcent2,aver_spec_m902/aver_spec_m902.max(),'r--')
    plt.plot(vcent,aver_spec_m90/aver_spec_m90.max(),'r', label='-90')
    plt.plot(vcent,aver_spec_m90/aver_spec_m90.max()-aver_spec_m902/aver_spec_m902.max(),'r:')
    plt.legend(prop={'size':7},loc=0)
    plt.subplot(235)
    plt.plot(vcent2,aver_spec_p902/aver_spec_p902.max(),'b--')
    plt.plot(vcent,aver_spec_p90/aver_spec_p90.max(),'b', label='+90')
    plt.plot(vcent,aver_spec_p90/aver_spec_p90.max()-aver_spec_p902/aver_spec_p902.max(),'b:')
    plt.legend(prop={'size':7},loc=0)
    plt.savefig(outdir+mole1+'_'+test1+'_'+ftype1+'-'+test2+'_'+ftype2+'_norm_spec_comparison_'+bmaj+'.pdf',dpi=100,bbox_inches='tight')
    plt.clf()

'''
# =======================================================================================
# Making teardrop plot of fiducial model
# =======================================================================================
# Radius range set # ====================================================================
r_min = 0.05;  r_max = 0.5; dr = 0.05   # in au arcsec
nr = int((r_max-r_min)/dr)
r_bin = np.arange(r_min,r_max+dr,dr)
rc = (r_bin[1:nr+1] + r_bin[0:nr])*0.5

# Set vel axis # ========================================================================
vref = (np.arange(cube.nv+1)-0.5)*cube.dv + cube.v0
vcent = np.average([vref[1:], vref[:-1]], axis=0)
aver_spec = np.zeros((nr,cube.nv)); std_spec = np.zeros((nr,cube.nv))
aver_spec_m = np.zeros((nr,cube.nv)); std_spec = np.zeros((nr,cube.nv))
PA_min=85; PA_max=95.0
# calculate averaged spectra # ==========================================================
for i in range(nr):
    r_in = r_bin[i]; r_out = r_bin[i+1]
    aver_spec[i,:], std_spec[i,:] = cube.get_azi_aver_spec(vref, inc, DPA, vmod, r_in, r_out, PA_min=PA_min, PA_max=PA_max, 
                          z0=z0, psi=psi, z1=z1, phi=phi, rt0=r_taper, qt0=q_taper)
PA_min=-95; PA_max=-85.0
# calculate averaged spectra # ==========================================================
for i in range(nr):
    r_in = r_bin[i]; r_out = r_bin[i+1]
    aver_spec_m[i,:], std_spec[i,:] = cube.get_azi_aver_spec(vref, inc, DPA, vmod, r_in, r_out, PA_min=PA_min, PA_max=PA_max, 
                          z0=z0, psi=psi, z1=z1, phi=phi, rt0=r_taper, qt0=q_taper)

# Plot teardrop figure # ================================================================
plt.figure(figsize=(10,6))
plt.pcolormesh(vcent,rc,aver_spec,norm=colors.LogNorm(vmin=1e-2*aver_spec.max(),vmax=aver_spec.max()),cmap='gist_heat',shading='gouraud',rasterized=True)
plt.axvline(x=5.0,color='m',ls='--')
plt.axvline(x=vsys,color='g',ls='--')
plt.axhline(y=120./160.,color='r',ls='--')
plt.axhline(y=260./160.,color='b',ls='--')
#plt.xlim(-2,10)
plt.ylim(r_min,r_max)
plt.xlabel(r'V$_{radio}$ (km/s)', fontsize=15)
plt.ylabel('R (arcsec)',fontsize=15)
#plt.legend(prop={'size':15},loc=1)
plt.tick_params(which='both',length=6,width=1.5,direction='in',color='w')
cbar = plt.colorbar() #ticks=[1e4,1e5,1e6,1e7])
cbar.set_label(r'$Jy\ beam^{-1}$', size=10)
plt.savefig('./teardrop/'+mole1+'_z0{:3.2f}_teardrop_'.format(z0)+test1+'_'+ftype+'_'+bmaj+'.pdf', bbox_inches='tight', pad_inches=0.1,dpi=100)
plt.clf()

plt.figure(figsize=(10,6))
#plt.axvline(x=5.0,color='m',ls='--')
#plt.axvline(x=vsys,color='g',ls='--')
plt.xlabel(r'V$_{radio}$ (km/s)', fontsize=15)
plt.ylabel('I (Jy/beam)',fontsize=15)
for i in range(nr):
    plt.plot(vcent, aver_spec[i,:],'k-')
    plt.plot(vcent, aver_spec_m[i,:],'r--')
plt.tick_params(which='both',length=6,width=1.5,direction='in',color='w')
plt.savefig('./teardrop/'+mole1+'_z0{:3.2f}_teardrop_spec_'.format(z0)+test1+'_'+ftype+'_'+bmaj+'.pdf', bbox_inches='tight', pad_inches=0.1,dpi=100)
plt.clf()

# =======================================================================================
# Making teardrop plot of comparing model
# =======================================================================================
# calculate averaged spectra # ==========================================================
aver_spec2 = np.zeros((nr,cube2.nv)); std_spec2 = np.zeros((nr,cube2.nv))
for i in range(nr):
    r_in = r_bin[i]; r_out = r_bin[i+1]
    aver_spec2[i,:], std_spec2[i,:] = cube2.get_azi_aver_spec(vref2, inc, DPA, vmod, r_in, r_out, PA_min=PA_min, PA_max=PA_max, 
                          z0=z0, psi=psi, z1=z1, phi=phi, rt0=r_taper, qt0=q_taper)

plt.figure(figsize=(10,6))
plt.pcolormesh(vcent,rc,aver_spec2,norm=colors.LogNorm(vmin=1e-2*aver_spec2.max(),vmax=aver_spec2.max()),cmap='gist_heat',shading='gouraud',rasterized=True)
plt.axvline(x=5.0,color='m',ls='--')
plt.axvline(x=vsys,color='g',ls='--')
plt.axhline(y=120./160.,color='r',ls='--')
plt.axhline(y=260./160.,color='b',ls='--')
#plt.xlim(-2,10)
#plt.ylim(0.0,2.5)
plt.xlabel(r'V$_{radio}$ (km/s)', fontsize=15)
plt.ylabel('R (arcsec)',fontsize=15)
#plt.legend(prop={'size':15},loc=1)
plt.tick_params(which='both',length=6,width=1.5,direction='in',color='w')
cbar = plt.colorbar() #ticks=[1e4,1e5,1e6,1e7])
cbar.set_label(r'$Jy\ beam^{-1}$', size=10)
plt.savefig('./teardrop/'+mole2+'_z0{:3.2f}_teardrop_'.format(z0)+test2+'_'+ftype+'_'+bmaj+'.pdf', bbox_inches='tight', pad_inches=0.1,dpi=100)
plt.clf()

plt.figure(figsize=(10,6))
#plt.axvline(x=5.0,color='m',ls='--')
#plt.axvline(x=vsys,color='g',ls='--')
plt.xlabel(r'V$_{radio}$ (km/s)', fontsize=15)
plt.ylabel('R (arcsec)',fontsize=15)
for i in range(nr):
    plt.plot(vcent, aver_spec2[i,:])
plt.tick_params(which='both',length=6,width=1.5,direction='in',color='w')
plt.savefig('./teardrop/'+mole2+'_z0{:3.2f}_teardrop_spec'.format(z0)+test2+'_'+ftype+'_'+bmaj+'.pdf', bbox_inches='tight', pad_inches=0.1,dpi=100)
plt.clf()

plt.figure(figsize=(10,6))
#plt.axvline(x=5.0,color='m',ls='--')
#plt.axvline(x=vsys,color='g',ls='--')
plt.xlim(-4,4)
plt.xlabel(r'V$_{radio}$ (km/s)', fontsize=15)
plt.ylabel(r'$\Delta$I (Jy/beam)',fontsize=15)
for i in range(nr):
    plt.plot(vcent, aver_spec[i,:] - aver_spec2[i,:])
#plt.plot(vcent, aver_spec[i,:])
#plt.plot(vcent, aver_spec2[i,:])
plt.tick_params(which='both',length=6,width=1.5,direction='in',color='w')
plt.savefig('./teardrop/'+mole+'_z0{:3.2f}_spec_diff_'.format(z0)+test1+'-'+test2+'_'+ftype+'_'+bmaj+'.pdf', bbox_inches='tight', pad_inches=0.1,dpi=100)
plt.clf()
'''


'''
# =======================================================================================
# Finding emitting surface by RADMC-3D tausurf
# =======================================================================================
mole = '12CO_2-1'
#fdir = '/Users/kimsj/Documents/RADMC-3D/radmc3d-2.0/RU_Lup_test/Automatics/Fin_script/tau1_surf_fits/'
fitsname = 'RULup_'+mole+'_fiducial_wind_sigma3_bmaj5_tau1.fits'
hdu = fits.open(fdir+fitsname)[0]   # Read the fits file: header + data
nx = hdu.header['NAXIS1']; ny = hdu.header['NAXIS2']; nv = hdu.header['NAXIS3']  # Set up axis lengths
bmaj = hdu.header['BMAJ']*3.6e3; bmin = hdu.header['BMIN']*3.6e3; bpa = hdu.header['BPA']
pixsize_x = abs(hdu.header['CDELT1']*3.6e3); pixsize_y = abs(hdu.header['CDELT2']*3.6e3)
if hdu.header['NAXIS'] == 4: data = hdu.data[0,:,:,:]      # Save the data part
if hdu.header['NAXIS'] == 3: data = hdu.data#[0,:,:,:]      # Save the data part
# Find the z_peak of tau = 1 surface
z_sky = np.zeros((nx,ny))
for i in range(nx):
    for j in range(ny):
        z_sky[i,j]=data[:,j,i].max()/au/d_pc
        if z_sky[i,j] == 0.0: z_sky[i,j] /= 0.0   # The 

# The disk inclined and rotated inversely. +y axis becomes near side & disk major axis rotates PA toward -y axis.
ux_sky, uy_sky, uz_sky = rotated_axis(inc, DPA)
print(ux_sky, uy_sky, uz_sky)
# Axes rotation from sky plane to disk plane
ux_disk, uy_disk, uz_disk = np.linalg.inv(np.array((ux_sky,uy_sky,uz_sky)))
print(ux_disk, uy_disk, uz_disk)
x=np.arange(-nx/2.,nx/2.)*pixsize_x#*d_pc # in au
y=np.arange(-ny/2.,ny/2.)*pixsize_y#*d_pc # in au
qq = np.meshgrid(x,y)
x_sky,y_sky = qq[0], qq[1]

xx_disk = vector_proj(x_sky, y_sky, z_sky, ux_disk)
yy_disk = vector_proj(x_sky, y_sky, z_sky, uy_disk)
zz_disk = vector_proj(x_sky, y_sky, z_sky, uz_disk)
zz_disk = np.where(np.isnan(zz_disk), 0.0, zz_disk)
rr_disk = np.hypot(xx_disk,yy_disk)

r = np.linspace(0.0,2.0,100)
z0 = 0.35; psi = 1.25
# Plotting the deprojected r-z points    
plt.figure(figsize=(10,6))
plt.scatter(rr_disk, zz_disk,marker='.',color='blue')
plt.plot(r, z0*np.power(r,psi),'r--',label='{:3.2f} r^{:3.2f}'.format(z0,psi))
#plt.plot(r, z0*(1+np.power(r/2.0,psi)),'b--',label='{:3.2f} r^{:3.2f}'.format(z0,psi))
plt.xlim(0,2.0)
#plt.ylim(0.0,0.5)
plt.xlabel('R (arcsec)', fontsize=15)
plt.ylabel('Z (arcsec)',fontsize=15)
plt.legend(prop={'size':7},loc=0)
plt.tick_params(which='both',length=6,width=1.5,direction='in')
plt.savefig('./teardrop/tausurf/'+mole+'_radmc3d_tausurf_zsurface.pdf',dpi=100,bbox_inches='tight')
#plt.show()
plt.clf()
'''

'''
# =======================================================================================
# Testing GoFish teardrop plot
# =======================================================================================
cube = imagecube(fdir+fitsname)
r_asc, velax, spectra, scatter  = cube.radial_spectra(inc=inc,PA=DPA,mstar=0.64,dist=160.0,
                                                      dr=0.05,z0=z0,psi=psi,z1=z1,phi=phi)
plt.figure(figsize=(10,6))
plt.pcolormesh(1e-3*velax,r_asc,spectra,norm=colors.LogNorm(vmin=1e-2*spectra.max(),vmax=spectra.max()),
               cmap='gist_heat',shading='gouraud',rasterized=True)
#plt.xlim(-1,2.4)
#plt.ylim(-2,-0.25)
plt.xlabel(r'V$_{radio}$ (km/s)', fontsize=15)
plt.ylabel('R (arcsec)',fontsize=15)
#plt.legend(prop={'size':15},loc=1)
plt.tick_params(which='both',length=6,width=1.5)
cbar = plt.colorbar() #ticks=[1e4,1e5,1e6,1e7])
cbar.set_label(r'$Jy\ beam^{-1}$', size=10)
plt.savefig('./'+mole+'_teardrop.pdf', bbox_inches='tight', pad_inches=0.1,dpi=100)
'''
'''
# =======================================================================================
# Finding emitting surface by disksurf (Teague et al. 2021; Pinte et al. 2018)
# =======================================================================================
mole = '12CO_2-1'
fdir = '/Users/kimsj/Documents/RU_Lup/Fin_fits/'
fitsname = mole+'_selfcal_matched_cube500.fits'
hdu = fits.open(fdir+fitsname)[0]   # Read the fits file: header + data
nx = hdu.header['NAXIS1']; ny = hdu.header['NAXIS2']; nv = hdu.header['NAXIS3']  # Set up axis lengths
bmaj = hdu.header['BMAJ']*3.6e3; bmin = hdu.header['BMIN']*3.6e3; bpa = hdu.header['BPA']
pixsize_x = abs(hdu.header['CDELT1']*3.6e3); pixsize_y = abs(hdu.header['CDELT2']*3.6e3)
cube = observation(fdir+fitsname)
chans = (50,105) # (50,105) for 2-1 lines / (30,70) for C18O 3-2 / (80,150) for 13CO 3-2 & CN 3-2
#cube.plot_channels(chans=chans)
surface = cube.get_emission_surface(inc=inc,PA=DPA,r_min=0.1,r_max=1.5,chans=chans,smooth=0.5)
rf_surf, zf_surf = [surface.r(side='front'), surface.z(side='front')]
rb_surf, zb_surf = [surface.r(side='back'), surface.z(side='back')]
plt.figure(figsize=(10,6))
plt.scatter(rf_surf, zf_surf,marker='o',color='blue')
plt.scatter(rb_surf, zb_surf,marker='o',color='red')
plt.xlim(0,1.5)
plt.ylim(-0.4,0.4)
plt.xlabel('R (arcsec)', fontsize=15)
plt.ylabel('Z (arcsec)',fontsize=15)
#plt.legend(prop={'size':15},loc=1)
plt.tick_params(which='both',length=6,width=1.5,direction='in')
#surface.plot_surface()
plt.savefig('./teardrop/'+mole+'_obs_disksurf_zsurface.pdf',dpi=100,bbox_inches='tight')
plt.clf()

chans = (70,110) # (70,105) #
cube.plot_peaks(surface=surface)
plt.savefig('./teardrop/'+mole+'_obs_disksurf_peaks.pdf',dpi=100,bbox_inches='tight')
'''

'''
# =======================================================================================
# Channel map for absorption feature at 5 km/s
# =======================================================================================
#from Kim_subroutines import *
mole = 'CN_3-2'
fdir = '/Users/kimsj/Documents/RU_Lup/Fin_fits/'
#fitsname = mole+'_selfcal_matched.fits'  
#outfig = './teardrop/'+mole+'_chmap_wide.pdf'
fitsname = mole+'_selfcal_clean_residual_chan50.fits'  
outfig = './teardrop/'+mole+'_chmap_residual.pdf'
cube = datacube(fdir+fitsname)
w = 4; h = 5
nchan = int(cube.nv/2.)-int(w*h/2.)
cube.plot_chmap(w, h, nchan=nchan, lim=6.0, vmax=0.01, vmin=-0.01, cmap='jet', ctext='k',
                figname=outfig, save=True)
'''