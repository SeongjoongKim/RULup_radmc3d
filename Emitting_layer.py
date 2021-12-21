import numpy as np
import matplotlib.colors as colors
from mpl_toolkits.mplot3d import axes3d
from matplotlib import pyplot as plt
from matplotlib import cm
from Model_setup_subroutines import *
import scipy.interpolate as inter
from scipy.ndimage import gaussian_filter1d
#from radmc3dPy.image import *
#from radmc3dPy.analyze import *
#from radmc3dPy.natconst import *

au  = 1.49598e13     # Astronomical Unit       [cm]
pc  = 3.08572e18     # Parsec                  [cm]
ms  = 1.98892e33     # Solar mass              [g]
rs  = 6.96e10        # Solar radius            [cm]
mp  = 1.6726e-24     # Mass of proton          [g]
GG  = 6.67408e-08    # Gravitational constant  [cm^3/g/s^2]
c = 2.99792458e10    # [cm/s]
#      Physical constants ----------------------------------
mu = 2.34
kb = 1.38e-16
NA = 6.02e23
mH = 1.0/NA
SB = 5.670374419e-5    # Stefan-Boltzmann constant [ erg cm-2 s-1 K-4]
h = 6.62607004e-27       # Planck function [ cm2 g s-1]

# ------------------------------------------------------------------------------------------
# Read grid file and set spherical & cylindrical coordinates
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

# ------------------------------------------------------------------------------------------
# Passive heating model at midplane
mstar    = 0.8*ms # Stempels et al 2002
rstar    = 1.64*rs  # Herczeg et al. 2005
tstar    = 3950.#  Stempels et al. 2002
lstar    = 4*np.pi*rstar**2*SB*tstar**4
r0 = 10.*au
t0 = ( 0.05*lstar/(8*np.pi*r0**2*SB) )**0.25  # 30*(10**0.16)**0.25 #  Adopting the DSHARP values (10**0.16*ls), Passive heating model
pltt = -0.5
tdust = t0 * (rr/r0)**pltt
cs       = np.sqrt(kb*tdust/(2.3*mp))   # Isothermal sound speed at midplane
omk      = np.sqrt(GG*mstar/rr**3)      # The Kepler angular frequency
# Disk parameters   ----------------------------------
hp       = cs/omk                      # The pressure scale height
hpr      = hp/rr                        # The dimensionless hp
settfact = 0.1               # Settling factor of the big grains
hp_biggr = hp*settfact
hpr_biggr= hpr*settfact
#  Gas disk ----------------------------------
sigmag0  = 1.5e3
rg0          = 1.
rge0        = 50.
plsigg      =  -1.0
#  Dust disk ----------------------------------
sigmad0 = 1.5e2      # dust surface density at rd0 au     [g/cm2]
plsig    = -1.0            # Powerlaw of the surface density
rd0     = 1              # Sigma_dust,0 at 5 au
rde0   = 50.           # exponential tail of dust profile

# Make the density model ----------------------------------
# Gas density
sigmag   = sigmag0*(rr/rg0/au)**plsigg #  power-law function
rhog = rho_normal(rr,zs,sigmag,hp)
# Dust density
sigmad  = sigmad0*(rr/rd0/au)**plsig*np.exp(-(rr/rde0/au)**4.0)
rhod = rho_normal(rr,zs,sigmad,hp_biggr)
rhos = rhog*1.0e-3
# Molecular fractional abundance
molecule_abun = fractional_abundance(rr,zs,'12CO')
NCO     = rhog * molecule_abun * dz/28.0/mp      #[cm2]
molecule_abun = fractional_abundance(rr,zs,'13CO')
N13CO     = rhog * molecule_abun * dz/29.0/mp      #[g/cm2]
molecule_abun = fractional_abundance(rr,zs,'C18O')
NC18O     = rhog * molecule_abun * dz/30.0/mp      #[g/cm2]
molecule_abun = fractional_abundance(rr,zs,'CN')
NCN     = rhog * molecule_abun * dz/26.0/mp      #[g/cm2]

# Read Temperature file ----------------------------------
test='DSHARP_H0.1_D0.1_S0.001_3e6'
fnameread      = 'Tdust_'+test+'.dat'
with open(fnameread,'r') as f:
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

# Interpolating the Tgas from spherical to cylindrical ----------------------------------
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


# Setting the line width at each pixel ----------------------------------
vturb    = 0.001*np.sqrt(GG*mstar/np.sqrt(rr**2+zs**2))   # [cm/s]
vther_CO    = therm_broadening(Tgas_cyl,28.0*mp)   # [cm/s]
vther_13CO    = therm_broadening(Tgas_cyl,29.0*mp)   # [cm/s]
vther_C18O    = therm_broadening(Tgas_cyl,30.0*mp)   # [cm/s]
vther_CN    = therm_broadening(Tgas_cyl,26.0*mp)   # [cm/s]
dV_CO = (vturb + vther_CO)#*0.5    # [cm/s]
dV_13CO = (vturb + vther_13CO)#*0.5    # [cm/s]
dV_C18O = (vturb + vther_C18O)#*0.5    # [cm/s]
dV_CN = (vturb + vther_CN)#*0.5    # [cm/s]

# Tau calculation (Goldsmith & Langer 1999, ApJ, 517, 209) ----------------------------------
level_CO, g_CO, nu_CO, Eu_CO, ul_CO = read_molecule_inp('co')
level_13CO, g_13CO, nu_13CO, Eu_13CO, ul_13CO = read_molecule_inp('13co')
level_C18O, g_C18O, nu_C18O, Eu_C18O, ul_C18O = read_molecule_inp('c18o')
level_CN, g_CN, nu_CN, Eu_CN,ul_CN = read_molecule_inp('cn')
Q_CO = Partition_func(g_CO, ul_CO, Eu_CO, Tgas_cyl)
Q_13CO = Partition_func(g_13CO, ul_13CO, Eu_13CO, Tgas_cyl)
Q_C18O = Partition_func(g_C18O, ul_C18O, Eu_C18O, Tgas_cyl)
Q_CN = Partition_func(g_CN, ul_CN, Eu_CN, Tgas_cyl)

# Line + Continuum optical depth   #  line_tau(J,nu_ref,E_u,mu,T,Q,N,dV) / target_dust_opacity(grain_size,obs_lamb)
tau_CO_21 = line_tau(2, nu_CO, Eu_CO, 0.11011 *10**(-18.0), Tgas_cyl, Q_CO, NCO, dV_CO) #+ target_dust_opacity('1mm',c/nu_CO[1]/1e9*1e4)*rhod*dz #+ target_dust_opacity('10um',c/nu_CO[1]/1e9*1e4)*rhos*dz
tau_13CO_21 = line_tau(2, nu_13CO, Eu_13CO, 0.11046 *10**(-18.0), Tgas_cyl, Q_13CO, N13CO, dV_13CO) #+ target_dust_opacity('1mm',c/nu_13CO[1]/1e9*1e4)*rhod*dz #+ target_dust_opacity('10um',c/nu_13CO[1]/1e9*1e4)*rhos*dz
tau_13CO_32 = line_tau(3, nu_13CO, Eu_13CO, 0.11046 *10**(-18.0), Tgas_cyl, Q_13CO, N13CO, dV_13CO)  #+ target_dust_opacity('1mm',c/nu_13CO[2]/1e9*1e4)*rhod*dz #+ target_dust_opacity('10um',c/nu_13CO[2]/1e9*1e4)*rhos*dz
tau_C18O_21 = line_tau(2, nu_C18O, Eu_C18O, 0.11079 *10**(-18.0), Tgas_cyl, Q_C18O, NC18O, dV_C18O)  #+ target_dust_opacity('1mm',c/nu_C18O[1]/1e9*1e4)*rhod*dz #+ target_dust_opacity('10um',c/nu_C18O[1]/1e9*1e4)*rhos*dz
tau_C18O_32 = line_tau(3, nu_C18O, Eu_C18O, 0.11079 *10**(-18.0), Tgas_cyl, Q_C18O, NC18O, dV_C18O)  #+ target_dust_opacity('1mm',c/nu_C18O[2]/1e9*1e4)*rhod*dz #+ target_dust_opacity('10um',c/nu_C18O[2]/1e9*1e4)*rhos*dz
tau_CN = line_tau(42, nu_CN, Eu_CN, 1.45*10**(-18.0), Tgas_cyl, Q_CN, NCN, dV_CN) + line_tau(44, nu_CN, Eu_CN, 1.45*10**(-18.0), Tgas_cyl, Q_CN, NCN, dV_CN) + line_tau(47, nu_CN, Eu_CN, 1.45*10**(-18.0), Tgas_cyl, Q_CN, NCN, dV_CN) #+ ( target_dust_opacity('1mm',c/nu_CN[41]/1e9*1e4)+target_dust_opacity('1mm',c/nu_CN[46]/1e9*1e4) )*rhod*dz #+ ( target_dust_opacity('10um',c/nu_CN[41]/1e9*1e4)+target_dust_opacity('10um',c/nu_CN[46]/1e9*1e4) )*rhos*dz

tau_d = target_dust_opacity('1mm',c/220.0/1e9*1e4)*rhod*dz + target_dust_opacity('10um',c/220.0/1e9*1e4)*rhos*dz

# Find tau = 1 layer
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

#"""
plt.figure(figsize=(10,6))
plt.pcolormesh(np.log10(rr/au),np.log10(zzr),Tgas_cyl,cmap='jet',norm=colors.LogNorm(vmin=10,vmax=250),shading='gouraud',rasterized=True)
plt.plot(np.log10(new_rc/au), np.log10(surf_CO_21), 'k', label=r'$\tau_{^{12}CO~2-1}$ = 1')
plt.plot(np.log10(new_rc/au), np.log10(surf_13CO_21), 'k--', label=r'$\tau_{^{13}CO~2-1}$ = 1')
#plt.plot(np.log10(new_rc/au), np.log10(surf_13CO_21), 'c--', label=r'$\tau_{^{13}CO~3-2}$ = 1')
plt.plot(np.log10(new_rc/au), np.log10(surf_C18O_21), 'k-.', label=r'$\tau_{C^{18}O~2-1}$ = 1')
#plt.plot(np.log10(new_rc/au), np.log10(surf_C18O_21), 'c-.', label=r'$\tau_{C^{18}O~3-2}$ = 1')
plt.plot(np.log10(new_rc/au), np.log10(surf_CN), 'k.', label=r'$\tau_{CN}$ = 1')
plt.plot(np.log10(new_rc/au), np.log10(surf_d), 'r', label=r'$\tau_{220GHz}$ = 1')
plt.plot(np.log10(new_rc/au), np.log10(conv_surf_CO_21), 'gray')
plt.xlim(-1,2.4)
plt.ylim(-2,-0.25)
plt.xlabel(r'log$_{10}$(r/1 AU)', fontsize=15)
plt.ylabel(r'log$_{10}$(z/r)',fontsize=15)
plt.legend(prop={'size':15},loc=0)
plt.tick_params(which='both',length=6,width=1.5)
#plt.xticks(np.log10([1,10,30,100,200]), ['1','10','30','100','200'])
plt.colorbar(ticks=[10,20,30,40,50,60,70,80,90,100,200])
#plt.savefig('Tdust_rz_small.pdf',dpi=100, bbox_inches='tight')
plt.savefig('Tgas_rz_cyl_interp.pdf',dpi=100,bbox_inches='tight')

# Radial T profile at the emitting surface ------------------------------------
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
plt.savefig('Tdust_radial_Emit_layer.pdf',dpi=100, bbox_inches='tight')
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
