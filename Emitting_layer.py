import numpy as np
import matplotlib.colors as colors
from mpl_toolkits.mplot3d import axes3d
from matplotlib import pyplot as plt
from matplotlib import cm
from Model_setup_subroutines import *
import scipy.interpolate as inter
from scipy.ndimage import gaussian_filter1d
from disksurf import observation
import os
#from radmc3dPy.image import *
#from radmc3dPy.analyze import *
#from radmc3dPy.natconst import *

#==============================================================
#      Physical constants
#==============================================================
au  = 1.49598e13     # Astronomical Unit       [cm]
pc  = 3.08572e18     # Parsec                  [cm]
ms  = 1.98892e33     # Solar mass              [g]
rs  = 6.96e10        # Solar radius            [cm]
mp  = 1.6726e-24     # Mass of proton          [g]
GG  = 6.67408e-08    # Gravitational constant  [cm^3/g/s^2]
c = 2.99792458e10    # [cm/s]
mu = 2.34
kb = 1.38e-16
NA = 6.02e23
mH = 1.0/NA
SB = 5.670374419e-5    # Stefan-Boltzmann constant [ erg cm-2 s-1 K-4]
h = 6.62607004e-27       # Planck function [ cm2 g s-1]

#==============================================================
# Set the model with/without disk wind
#==============================================================
wind='T' # Set wind model
test='fiducial_wind_rot_wind'  # file name of Tdust_test.dat
mole = 'CN_3-2'; bmaj = 'bmaj5'  # Set molecule and resolution
tname = 'fiducial_wind_rot_wind'          # Set file name for test

#==============================================================
# Read grid file and set spherical & cylindrical coordinates
#==============================================================
fnameread      = 'amr_grid.inp'
with open(fnameread,'r') as f:
    s1         = f.readline()
    s2         = f.readline()
    s3         = f.readline()
    s4         = f.readline()
    s5      = f.readline()
    ngrid   = f.readline()
    nr, ntheta, nphi  = int(ngrid.split()[0]), int(ngrid.split()[1]), int(ngrid.split()[2])
    grids = np.zeros(nr+ntheta+nphi+3)
    for ilam in range(len(grids)):
        grids[ilam]  = f.readline()


ri = grids[0:nr+1]; thetai = grids[nr+1:nr+ntheta+2]; phii = grids[nr+ntheta+2:-1]
rc       = 0.5 * ( ri[0:nr] + ri[1:nr+1] )
thetac   = 0.5 * ( thetai[0:ntheta] + thetai[1:ntheta+1] )
phic     = 0.5 * ( phii[0:nphi] + phii[1:nphi+1] )
qq       = np.meshgrid(rc,thetac,indexing='ij')
rr       = qq[0]    # cm
tt       = qq[1]     # rad
zzr       = np.pi/2.e0 - qq[1]     # rad in spherical
zs       = rr*np.tan(zzr)          # cm in cylindrical
dz       = rr*(thetac[1:ntheta]-thetac[0:ntheta-1]).mean()   # cm in cylindrical

#==============================================================
# Passive heating model at midplane
#==============================================================
mstar    = 0.63*ms # Stempels et al 2002
rstar    = 2.42*rs  # Herczeg et al. 2005
tstar    = 4073.#  Stempels et al. 2002
lstar    = 4*np.pi*rstar**2*SB*tstar**4
r0 = 10.*au
t0 = ( 0.05*lstar/(8*np.pi*r0**2*SB) )**0.25  # 30*(10**0.16)**0.25 #  Adopting the DSHARP values (10**0.16*ls), Passive heating model
pltt = -0.5
tdust = t0 * (rr/r0)**pltt
cs       = np.sqrt(kb*tdust/(2.3*mp))   # Isothermal sound speed at midplane
omk      = np.sqrt(GG*mstar/rr**3)      # The Kepler angular frequency
#==============================================================
# Disk parameters
#==============================================================
hp       = cs/omk                      # The pressure scale height
hpr      = hp/rr                        # The dimensionless hp
settfact = 0.1               # Settling factor of the big grains
hp_biggr = hp*settfact
hpr_biggr= hpr*settfact
#  Gas disk
sigmag0  = 1.5e3
rg0          = 1.
rge0        = 50.
plsigg      =  -1.0
#  Dust disk
sigmad0 = 1.5e2      # dust surface density at rd0 au     [g/cm2]
plsig    = -1.0            # Powerlaw of the surface density
rd0     = 1              # Sigma_dust,0 at 1 au
rde0   = 50.           # exponential tail of dust profile

#==============================================================
# Set the inner edge conditions
#==============================================================
r_inedge = 0.1*au; t_inedge = t0*(r_inedge/r0)**pltt
zb_inedge = 4.0*np.sqrt(kb*t_inedge/(2.3*mp))/np.sqrt(GG*mstar/r_inedge**3) - 0.1*au

a=0.1; b=1.0 # Set the disk inner edge (0.1 au) and tapering radius (1 au)
s_to_g_ratio = 1.0e-3  # Small grain to gas mass ratio. The default is set as 0.001
#==============================================================
# Make the density model
#==============================================================
sigmag = np.zeros_like(rr)
sigmad = np.zeros_like(rr)
for i in range(nr):
    for j in range(ntheta):
        if rr[i,j]>=1*au:
            sigmag[i,j]   = sigmag0*(rr[i,j]/rg0/au)**plsigg #  power-law function
            sigmad[i,j]   = sigmad0*(rr[i,j]/rd0/au)**plsig*np.exp(-(rr[i,j]/rde0/au)**2.0) #  power-law function
        else:
            sigmag[i,j]   = sigmag0*(1.0-(rr[i,j]/a/au))*(1.0-(b/a))**(-1) #  power-law function
            sigmad[i,j]   = sigmad0*(1.0-(rr[i,j]/a/au))*(1.0-(b/a))**(-1) #  power-law function
sigmag[np.where(rr<0.1*au)] = 0.
sigmad[np.where(rr<0.1*au)] = 0.
#==============================================================
beta_0, Cw, alpha_B = [1.e4, 1.0e-5, -2.0] # Set the beta0, Cw, and alpha_B values.
if wind == 'F':
    # Gas density
    #sigmag   = sigmag0*(rs/rg0/au)**plsigg #  power-law function
    rhog = rho_normal(rr,zs,sigmag,hp)
    # Dust density
    #sigmad  = sigmad0*(rs/rd0/au)**plsig*np.exp(-(rs/rde0/au)**2.0)
    rhod = rho_normal(rr,zs,sigmad,hp_biggr)
    rhos = rhog*s_to_g_ratio
elif wind == 'T':
    # Gas density
    #sigmag   = sigmag0*(rs/rg0/au)**plsigg #  power-law function
    P_mid = sigmag/np.sqrt(2.e0*np.pi)/hp *cs**2   # Equation (18) of Bai 2016, ApJ, 821, 80
    B_mid = np.sqrt((8*np.pi*P_mid)/beta_0 )             # Equation (18) of Bai 2016, ApJ, 821, 80
    z0 = 4.0*hp  #X*rs**(1.5+0.5*pltt)  # wind base
    rhog = np.zeros_like(rr)   # [g/cm3] in Spherical (r,theta)
    Bp = np.zeros_like(rr)
    vz = np.zeros_like(rr)
    # Calculate rho_gas, Bp, and vz: Bp and vz are values along cylindrical z axis.
    for i in range(0,nr):
        for j in range(0,ntheta):
            if (zs[i,j]<=z0[i,j]):
                rhog[i,j] = ( sigmag[i,j] / (np.sqrt(2.e0*np.pi)*hp[i,j]) ) * np.exp(-zs[i,j]**2/hp[i,j]**2/2.e0)  #[g/cm3]
                Bp[i,j] = B_mid[i,j]
                vz[i,j] = Cw*sigmag[i,j]*omk[i,j]/rhog[i,j]   # Suzuki et al. 2010, ApJ, 718, 1289
            else:
                if (zs[i,j]-rr[i,j] > zb_inedge):
                    rhog[i,j] = 0.
                    Bp[i,j] = 0.
                    vz[i,j] = 0.
                else:
                    R0 = calculate_R0(rr[i,j],zs[i,j],t0,r0,pltt,mstar)   #z0[i,j]-zs[i,j]+rs[i,j]
                    if R0<1*au:
                        sigmag_R0 = sigmag0*(1.0-(R0/a/au))*(1.0-(b/a))**(-1)
                    else:
                        sigmag_R0 = sigmag0*(R0/rg0/au)**plsigg
                    t_R0 = t0 * (R0/r0)**pltt
                    cs_R0 = np.sqrt(kb*t_R0/(2.3*mp))
                    omk_R0 = np.sqrt(GG*mstar/R0**3)
                    hp_R0 = cs_R0 / omk_R0
                    rhog_R0 = sigmag_R0 / np.sqrt(2.e0*np.pi)/hp_R0
                    P_mid_R0 = rhog_R0*cs_R0**2
                    B_mid_R0 = np.sqrt((8*np.pi*P_mid_R0)/beta_0)
                    rhog[i,j] = rhog_R0 * np.exp(-8.0) * np.exp(-(cs_R0/R0/omk_R0)**(-0.6) * np.sqrt((zs[i,j]-(4*hp_R0))/R0 ) )
                    Bp[i,j] = B_mid_R0* (rr[i,j]/R0)**alpha_B #*np.cos(45.*np.pi/180.)  # Bai 2016, ApJ, 818,152
                    vz[i,j] = Cw*sigmag_R0*omk_R0*Bp[i,j]/B_mid_R0/rhog[i,j] #*np.cos(45.*np.pi/180.)
    # Dust density
    #sigmad  = sigmad0*(rs/rd0/au)**plsig*np.exp(-(rs/rde0/au)**2.0)
    rhod = rho_normal(rr,zs,sigmad,hp_biggr)
    rhos = rhog*s_to_g_ratio
else:
    raise ValueError("Unknown wind model")

#==============================================================
# Molecular fractional abundance
#==============================================================
molecule_abun = fractional_abundance(rr,zs,'12CO')
NCO     = rhog * molecule_abun * dz/28.0/mp      #[cm2]
molecule_abun = fractional_abundance(rr,zs,'13CO')
N13CO     = rhog * molecule_abun * dz/29.0/mp      #[g/cm2]
molecule_abun = fractional_abundance(rr,zs,'C18O')
NC18O     = rhog * molecule_abun * dz/30.0/mp      #[g/cm2]
molecule_abun = fractional_abundance(rr,zs,'CN')
NCN     = rhog * molecule_abun * dz/26.0/mp      #[g/cm2]

#==============================================================
# Read Temperature file
#==============================================================
fdir = '/Users/kimsj/Documents/RADMC-3D/radmc3d-2.0/RU_Lup_test/Automatics/Fin_script/fiducial_wind/'
fnameread      = 'Tdust_'+test+'.dat'
with open(fdir+fnameread,'r') as f:
    s         = f.readline()
    im_n      = int(f.readline())
    npop   = int(f.readline())
    temp = np.zeros(im_n*npop)
    for ilam in range(npop):
        for iy in range(im_n):
            temp[iy+(im_n*ilam)]  = f.readline()


temp=temp.reshape((npop,ntheta,nr))
temp_smlgr=temp[0,:,:].T
temp_biggr=temp[1,:,:].T

#==============================================================
# Interpolating the Tgas from spherical to cylindrical
#==============================================================
Tgas_cyl = np.zeros_like(temp_smlgr)
for i in range(0,ntheta):
    Tgas_sph = temp_smlgr[:,i]
    intp_Tgas = inter.interp1d(rc,Tgas_sph,kind='cubic')
    for j in range(0,nr):
        dis = np.sqrt(rr[j,i]**2+zs[j,i]**2)
        if (dis <= rc[-1]):
            Tgas_cyl[j,i] = intp_Tgas(dis)
        else:
            Tgas_cyl[j,i] = temp_smlgr[j,i]

#==============================================================
# Setting the line width at each pixel
#==============================================================
vturb    = 0.001*np.sqrt(GG*mstar/np.sqrt(rr**2+zs**2))   # [cm/s]
vther_CO    = therm_broadening(Tgas_cyl,28.0*mp)   # [cm/s]
vther_13CO    = therm_broadening(Tgas_cyl,29.0*mp)   # [cm/s]
vther_C18O    = therm_broadening(Tgas_cyl,30.0*mp)   # [cm/s]
vther_CN    = therm_broadening(Tgas_cyl,26.0*mp)   # [cm/s]
dV_CO = (vturb + vther_CO)#*0.5    # [cm/s]
dV_13CO = (vturb + vther_13CO)#*0.5    # [cm/s]
dV_C18O = (vturb + vther_C18O)#*0.5    # [cm/s]
dV_CN = (vturb + vther_CN)#*0.5    # [cm/s]

#==============================================================
# Tau calculation (Goldsmith & Langer 1999, ApJ, 517, 209)
#==============================================================
level_CO, g_CO, nu_CO, Eu_CO, ul_CO = read_molecule_inp('co')
level_13CO, g_13CO, nu_13CO, Eu_13CO, ul_13CO = read_molecule_inp('13co')
level_C18O, g_C18O, nu_C18O, Eu_C18O, ul_C18O = read_molecule_inp('c18o')
level_CN, g_CN, nu_CN, Eu_CN,ul_CN = read_molecule_inp('cn')
Q_CO = Partition_func(g_CO, ul_CO, Eu_CO, Tgas_cyl)
Q_13CO = Partition_func(g_13CO, ul_13CO, Eu_13CO, Tgas_cyl)
Q_C18O = Partition_func(g_C18O, ul_C18O, Eu_C18O, Tgas_cyl)
Q_CN = Partition_func(g_CN, ul_CN, Eu_CN, Tgas_cyl)

#==============================================================
# Line + Continuum optical depth   #  line_tau(J,nu_ref,E_u,mu,T,Q,N,dV) / target_dust_opacity(grain_size,obs_lamb)
#==============================================================
tau_CO_21 = line_tau(2, nu_CO, Eu_CO, 0.11011 *10**(-18.0), Tgas_cyl, Q_CO, NCO, dV_CO) + target_dust_opacity('1mm',c/nu_CO[1]/1e9*1e4)*rhod*dz #+ target_dust_opacity('10um',c/nu_CO[1]/1e9*1e4)*rhos*dz
tau_13CO_21 = line_tau(2, nu_13CO, Eu_13CO, 0.11046 *10**(-18.0), Tgas_cyl, Q_13CO, N13CO, dV_13CO) + target_dust_opacity('1mm',c/nu_13CO[1]/1e9*1e4)*rhod*dz #+ target_dust_opacity('10um',c/nu_13CO[1]/1e9*1e4)*rhos*dz
tau_13CO_32 = line_tau(3, nu_13CO, Eu_13CO, 0.11046 *10**(-18.0), Tgas_cyl, Q_13CO, N13CO, dV_13CO)  + target_dust_opacity('1mm',c/nu_13CO[2]/1e9*1e4)*rhod*dz #+ target_dust_opacity('10um',c/nu_13CO[2]/1e9*1e4)*rhos*dz
tau_C18O_21 = line_tau(2, nu_C18O, Eu_C18O, 0.11079 *10**(-18.0), Tgas_cyl, Q_C18O, NC18O, dV_C18O)  + target_dust_opacity('1mm',c/nu_C18O[1]/1e9*1e4)*rhod*dz #+ target_dust_opacity('10um',c/nu_C18O[1]/1e9*1e4)*rhos*dz
tau_C18O_32 = line_tau(3, nu_C18O, Eu_C18O, 0.11079 *10**(-18.0), Tgas_cyl, Q_C18O, NC18O, dV_C18O)  + target_dust_opacity('1mm',c/nu_C18O[2]/1e9*1e4)*rhod*dz #+ target_dust_opacity('10um',c/nu_C18O[2]/1e9*1e4)*rhos*dz
tau_CN = line_tau(42, nu_CN, Eu_CN, 1.45*10**(-18.0), Tgas_cyl, Q_CN, NCN, dV_CN) + line_tau(44, nu_CN, Eu_CN, 1.45*10**(-18.0), Tgas_cyl, Q_CN, NCN, dV_CN) + line_tau(47, nu_CN, Eu_CN, 1.45*10**(-18.0), Tgas_cyl, Q_CN, NCN, dV_CN) + ( target_dust_opacity('1mm',c/nu_CN[41]/1e9*1e4)+target_dust_opacity('1mm',c/nu_CN[46]/1e9*1e4) )*rhod*dz #+ ( target_dust_opacity('10um',c/nu_CN[41]/1e9*1e4)+target_dust_opacity('10um',c/nu_CN[46]/1e9*1e4) )*rhos*dz
tau_d = target_dust_opacity('1mm',c/220.0/1e9*1e4)*rhod*dz + target_dust_opacity('10um',c/220.0/1e9*1e4)*rhos*dz

#==============================================================
# Find tau = 1 layer
#==============================================================
new_rc, surf_CO_21, z_co21 = emit_surface(rc,zs,tau_CO_21,nr)
new_rc, surf_13CO_21, z_13co21 = emit_surface(rc,zs,tau_13CO_21,nr)
new_rc, surf_13CO_32, z_13co32 = emit_surface(rc,zs,tau_13CO_32,nr)
new_rc, surf_C18O_21, z_c18o21 = emit_surface(rc,zs,tau_C18O_21,nr)
new_rc, surf_C18O_32, z_c18o32 = emit_surface(rc,zs,tau_C18O_32,nr)
new_rc, surf_CN, z_cn = emit_surface(rc,zs,tau_CN,nr)
new_rc, surf_d, z_cn = emit_surface(rc,zs,tau_d,nr)
conv_surf_CO_21 = gaussian_filter1d(surf_CN,20)

temp_12CO = np.zeros(len(rc))
temp_13CO = np.zeros(len(rc))
temp_C18O = np.zeros(len(rc))
temp_CN = np.zeros(len(rc))
for i in range(len(rc)):
    temp_12CO[i] = Tgas_cyl[i,int(z_co21[i])]
    temp_13CO[i] = Tgas_cyl[i,int(z_13co21[i])]
    temp_C18O[i] = Tgas_cyl[i,int(z_c18o21[i])]
    temp_CN[i] = Tgas_cyl[i,int(z_cn[i])]

#==============================================================
# Find emitting surface by disksurf package (Pinte et al. 2018; Teague et al. 2021)
#==============================================================
distance = 160.0
inc = 25.0; DPA = 121.0
fdir2 = '/Users/kimsj/Documents/RADMC-3D/radmc3d-2.0/RU_Lup_test/Automatics/Fin_script/fiducial_wind/'
windname = 'RULup_'+mole+'_'+tname+'_'+bmaj+'.fits'
if not os.path.exists(fdir2+windname):
    windname = 'RULup_'+mole+'_fiducial_wind_'+bmaj+'.fits'#
cube = observation(fdir2+windname)
chans = [85,114]
#cube.plot_channels(chans=chans)
surface = cube.get_emission_surface(inc=inc,PA=DPA,r_min=0,r_max=1.5,chans=chans,smooth=0.5)
#surface.plot_surface()
cube.plot_peaks(surface=surface)
plt.savefig('RULup_'+mole+'_'+tname+'_disksurf_peaks.pdf',dpi=100,bbox_inches='tight')
r_surf, z_surf = [surface.r(side='front')* distance, surface.z(side='front')* distance]

#"""
#==============================================================
# Plotting emitting layers in disk r-z plane
#==============================================================
plt.figure(figsize=(10,6))
im = plt.pcolormesh(np.log10(rr/au),np.log10(zzr),Tgas_cyl,cmap='jet',norm=colors.LogNorm(vmin=10,vmax=250),shading='gouraud',rasterized=True)
plt.plot(np.log10(new_rc/au), np.log10(surf_CO_21), 'k', label=r'$\tau_{^{12}CO~2-1}$ = 1')
plt.plot(np.log10(new_rc/au), np.log10(surf_13CO_21), 'k--', label=r'$\tau_{^{13}CO~2-1}$ = 1')
#plt.plot(np.log10(new_rc/au), np.log10(surf_13CO_21), 'c--', label=r'$\tau_{^{13}CO~3-2}$ = 1')
plt.plot(np.log10(new_rc/au), np.log10(surf_C18O_21), 'k-.', label=r'$\tau_{C^{18}O~2-1}$ = 1')
#plt.plot(np.log10(new_rc/au), np.log10(surf_C18O_21), 'c-.', label=r'$\tau_{C^{18}O~3-2}$ = 1')
plt.plot(np.log10(new_rc/au), np.log10(surf_CN), 'k:', label=r'$\tau_{CN}$ = 1')
plt.plot(np.log10(new_rc/au), np.log10(surf_d), 'gray', label=r'$\tau_{220GHz}$ = 1')
#plt.plot(np.log10(new_rc/au), np.log10(conv_surf_CO_21), 'gray')
plt.scatter(np.log10(r_surf),np.log10(z_surf/r_surf),s=5,c='m',label='disksurf')
plt.xlim(0.5,2.5)
plt.ylim(-2,-0.25)
plt.xlabel(r'log$_{10}$(r/1 AU)', fontsize=15)
plt.ylabel(r'log$_{10}$(z/r)',fontsize=15)
plt.legend(prop={'size':12},loc=2)
plt.tick_params(which='both',length=6,width=1.5)
#plt.xticks(np.log10([1,10,30,100,200]), ['1','10','30','100','200'])
cbar = plt.colorbar(im, ticks=[20,40,60,80,100,120,140,200])
cbar.ax.set_yticklabels([20,40,60,80,100,120,140,200])
#plt.savefig('Tdust_rz_small.pdf',dpi=100, bbox_inches='tight')
plt.savefig('Tgas_rz_cyl_interp_'+mole+'_'+tname+'_layer.pdf',dpi=100,bbox_inches='tight')
#==============================================================
# Radial T profile at the emitting surface
#==============================================================
plt.figure(figsize=(10,6))
plt.plot(rc/au,temp_12CO,'k',lw=2,label=r'T$_{\rm 12CO}$')
plt.plot(rc/au,temp_13CO,'k--',lw=2,label=r'T$_{\rm 13CO}$')
plt.plot(rc/au,temp_C18O,'k-.',lw=2,label=r'T$_{\rm C18O}$')
plt.plot(rc/au,temp_CN,'k.',lw=2,label=r'T$_{\rm CN}$')
plt.xlim(1,300)
plt.ylim(10,100)
plt.xlabel(r'log$_{10}$(r/1 AU)', fontsize=15)
plt.ylabel(r'T$_{gas}$',fontsize=15)
plt.legend(prop={'size':15},loc=0)
plt.tick_params(which='both',length=6,width=1.5)
#plt.colorbar(ticks=[10,20,30,40,50,60,70,80,90,100,200])
plt.savefig('Tdust_radial_'+tname+'_Emit_layer.pdf',dpi=100, bbox_inches='tight')
#plt.savefig('temp_rz_big.pdf',dpi=100, bbox_inches='tight')
#"""

"""
# Plot T distribution in r,z plane ------------------------------------
plt.figure(figsize=(10,6))
plt.pcolormesh(np.log10(rr/au),np.log10(zzr),temp_biggr,cmap='jet',norm=colors.LogNorm(vmin=10,vmax=250),shading='gouraud',rasterized=True)
plt.xlim(0,2.4)
plt.ylim(-2,-0.25)
plt.xlabel(r'log$_{10}$(r/1 AU)', fontsize=15)
plt.ylabel(r'log$_{10}$(z/r)',fontsize=15)
#plt.legend(prop={'size':15},loc=1)
plt.tick_params(which='both',length=6,width=1.5)
plt.colorbar(ticks=[10,20,30,40,50,60,70,80,90,100,200])
#plt.savefig('Tdust_rz_small.pdf',dpi=100, bbox_inches='tight')
plt.savefig('Tdust_rz_'+test+'.pdf',dpi=100,bbox_inches='tight')


# Plot T distribution in r,z plane ------------------------------------
plt.figure(figsize=(10,6))
plt.pcolormesh(np.log10(rr/au),np.log10(zzr),temp_smlgr,cmap='jet',norm=colors.LogNorm(vmin=10,vmax=250),shading='gouraud',rasterized=True)
plt.xlim(0,2.4)
plt.ylim(-2,-0.25)
plt.xlabel(r'log$_{10}$(r/1 AU)', fontsize=15)
plt.ylabel(r'log$_{10}$(z/r)',fontsize=15)
#plt.legend(prop={'size':15},loc=1)
plt.tick_params(which='both',length=6,width=1.5)
plt.colorbar(ticks=[10,20,30,40,50,60,70,80,90,100,200])
#plt.savefig('Tdust_rz_small.pdf',dpi=100, bbox_inches='tight')
plt.savefig('Tgas_rz_'+test+'.pdf',dpi=100,bbox_inches='tight')


# Radial T profile at the midplane ------------------------------------
plt.figure(figsize=(10,6))
plt.plot(rr[:,-1]/au,temp_biggr[:,-1],'k',lw=2,label=r'Midplane T$_{\rm dust}$')
plt.plot(rr[:,-1]/au,temp_smlgr[:,-1],'k--',lw=2,label=r'Midplane T$_{\rm gas}$')
plt.plot(rc/au,tdust,'r',lw=2,label=r'Midplane T$_{\rm model}$')
plt.xlim(10,200)
plt.ylim(5,50)
plt.xlabel(r'log$_{10}$(r/1 AU)', fontsize=15)
plt.ylabel(r'T$_{dust}$',fontsize=15)
plt.legend(prop={'size':15},loc=0)
plt.tick_params(which='both',length=6,width=1.5)
#plt.colorbar(ticks=[10,20,30,40,50,60,70,80,90,100,200])
plt.savefig('Tdust_radial_mid_'+mole[i]+'.pdf',dpi=100, bbox_inches='tight')
#plt.savefig('temp_rz_big.pdf',dpi=100, bbox_inches='tight')
"""


