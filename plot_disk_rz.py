import numpy as np
import matplotlib.colors as colors
from mpl_toolkits.mplot3d import axes3d
from matplotlib import pyplot as plt
from matplotlib import cm
from Model_setup_subroutines import *
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-file', default='None', type=str, help='Input test title')
args = parser.parse_args()
test = args.file

au  = 1.49598e13     # Astronomical Unit       [cm]
ms  = 1.98892e33     # Solar mass              [g]
rs  = 6.96e10        # Solar radius            [cm]

#------------------------------------------------------------------------------------------------------------
fnameread      = 'amr_grid.inp'
nr, ntheta, nphi, grids = read_grid_inp(fnameread)
ri = grids[0:nr+1]; thetai = grids[nr+1:nr+ntheta+2]; phii = grids[nr+ntheta+2:-1]
rc       = 0.5 * ( ri[0:nr] + ri[1:nr+1] )
thetac   = 0.5 * ( thetai[0:ntheta] + thetai[1:ntheta+1] )
phic     = 0.5 * ( phii[0:nphi] + phii[1:nphi+1] )
qq       = np.meshgrid(rc,thetac,indexing='ij')
rr       = qq[0]    # cm
tt       = qq[1]     # rad
zzr       = np.pi/2.e0 - qq[1]     # rad

#------------------------------------------------------------------------------------------------------------
# Plot for the density distribution ------------------------------------
im_n, npop, temp = read_dens_inp('dust_density.inp')
temp=temp.reshape((npop,ntheta,nr))
rhos=temp[0,:,:].T
rhod=temp[1,:,:].T

im_n, npop, temp = read_dens_inp('gas_density.inp')
temp=temp.reshape((npop,ntheta,nr))
rhog=temp[0,:,:].T

plt.figure(figsize=(10,6))
plt.pcolormesh(np.log10(rr/au),np.log10(zzr),rhod,cmap='jet',norm=colors.LogNorm(vmin=1e-17,vmax=1e-10),shading='gouraud',rasterized=True)
#plt.pcolormesh(np.log10(rr/au),np.log10(zzr),rhod_biggr,cmap='gist_heat',norm=colors.LogNorm(vmin=3e-13,vmax=3e-10),shading='gouraud')
#plt.plot(np.log10(a_2/au),layer12,'k',lw=2,label=r'$\tau_{\rm 12CO}$=1')
#plt.plot(np.log10(a_2/au),layer13,'k--',lw=2,label=r'$\tau_{\rm 13CO}$=1')
plt.xlim(-1,2.4)
plt.ylim(-2,-0.25)
plt.xlabel(r'log$_{10}$(r/1 AU)', fontsize=15)
plt.ylabel(r'log$_{10}$(z/r)',fontsize=15)
#plt.legend(prop={'size':15},loc=1)
plt.tick_params(which='both',length=6,width=1.5)
plt.colorbar()
plt.savefig('Density_large_'+test+'_rz.pdf',dpi=100, bbox_inches='tight')

plt.figure(figsize=(10,6))
plt.pcolormesh(np.log10(rr/au),np.log10(zzr),rhos,cmap='jet',norm=colors.LogNorm(vmin=1e-17,vmax=1e-10),shading='gouraud',rasterized=True)
#plt.pcolormesh(np.log10(rr/au),np.log10(zzr),rhod_biggr,cmap='gist_heat',norm=colors.LogNorm(vmin=3e-13,vmax=3e-10),shading='gouraud')
#plt.plot(np.log10(a_2/au),layer12,'k',lw=2,label=r'$\tau_{\rm 12CO}$=1')
#plt.plot(np.log10(a_2/au),layer13,'k--',lw=2,label=r'$\tau_{\rm 13CO}$=1')
plt.xlim(-1,2.4)
plt.ylim(-2,-0.25)
plt.xlabel(r'log$_{10}$(r/1 AU)', fontsize=15)
plt.ylabel(r'log$_{10}$(z/r)',fontsize=15)
#plt.legend(prop={'size':15},loc=1)
plt.tick_params(which='both',length=6,width=1.5)
plt.colorbar()
plt.savefig('Density_small_'+test+'_rz.pdf',dpi=100, bbox_inches='tight')

plt.figure(figsize=(10,6))
plt.pcolormesh(np.log10(rr/au),np.log10(zzr),rhog,cmap='jet',norm=colors.LogNorm(vmin=1e-16,vmax=1e-9),shading='gouraud',rasterized=True)
#plt.pcolormesh(np.log10(rr/au),np.log10(zzr),rhod_biggr,cmap='gist_heat',norm=colors.LogNorm(vmin=3e-13,vmax=3e-10),shading='gouraud')
#plt.plot(np.log10(a_2/au),layer12,'k',lw=2,label=r'$\tau_{\rm 12CO}$=1')
#plt.plot(np.log10(a_2/au),layer13,'k--',lw=2,label=r'$\tau_{\rm 13CO}$=1')
plt.xlim(-1,2.4)
plt.ylim(-2,-0.25)
plt.xlabel(r'log$_{10}$(r/1 AU)', fontsize=15)
plt.ylabel(r'log$_{10}$(z/r)',fontsize=15)
#plt.legend(prop={'size':15},loc=1)
plt.tick_params(which='both',length=6,width=1.5)
plt.colorbar()
plt.savefig('Density_gas_'+test+'_rz.pdf',dpi=100, bbox_inches='tight')

plt.figure(figsize=(10,6))
plt.plot(np.log10(rr[:,0]/au),np.log10(rhod[:,-1]),'k',lw=2,label=r'Midplane $\rho_{\rm dust}$')
plt.plot(np.log10(rr[:,0]/au),np.log10(rhog[:,-1]),'k--',lw=2,label=r'Midplane $\rho_{\rm gas}$')
#plt.pcolormesh(np.log10(rr/au),np.log10(zzr),temp_biggr,cmap='jet',norm=colors.LogNorm(vmin=10,vmax=250),shading='gouraud')
#plt.plot(np.log10(a_2/au),layer12,'k',lw=2,label=r'$\tau_{\rm 12CO}$=1')
#plt.plot(np.log10(a_2/au),layer13,'k--',lw=2,label=r'$\tau_{\rm 13CO}$=1')
plt.xlim(-1,2.4)
#plt.ylim(-20,5)
plt.xlabel(r'log$_{10}$(r/1 AU)', fontsize=15)
plt.ylabel(r'$\rho$',fontsize=15)
plt.legend(prop={'size':15},loc=0)
plt.tick_params(which='both',length=6,width=1.5)
#plt.colorbar(ticks=[10,20,30,40,50,60,70,80,90,100,200])
plt.savefig('Density_radial_'+test+'_mid.pdf',dpi=100, bbox_inches='tight')


#------------------------------------------------------------------------------------------------------------
# Read veolcity field files   ------------------------------------
vr, vtheta, vphi = read_vel_input('gas_velocity.inp')
vr=vr.reshape((ntheta,nr)).T*1e-5
vtheta=vtheta.reshape((ntheta,nr)).T*1e-5
vphi=vphi.reshape((ntheta,nr)).T*1e-5
#print(rhod.shape)

# Plotting radial velocity field  ------------------------------------
plt.figure(figsize=(10,6))
plt.pcolormesh(np.log10(rr/au),np.log10(zzr),vr,norm=colors.LogNorm(vmin=1e-2,vmax=1),cmap='jet',shading='gouraud',rasterized=True)
#plt.pcolormesh(np.log10(rr/au),np.log10(zzr),rhod_biggr,cmap='gist_heat',norm=colors.LogNorm(vmin=3e-13,vmax=3e-10),shading='gouraud')
#plt.plot(np.log10(a_2/au),layer12,'k',lw=2,label=r'$\tau_{\rm 12CO}$=1')
#plt.plot(np.log10(a_2/au),layer13,'k--',lw=2,label=r'$\tau_{\rm 13CO}$=1')
plt.xlim(-1,2.4)
plt.ylim(-2,-0.25)
plt.xlabel(r'log$_{10}$(r/1 AU)', fontsize=15)
plt.ylabel(r'log$_{10}$(z/r)',fontsize=15)
#plt.legend(prop={'size':15},loc=1)
plt.tick_params(which='both',length=6,width=1.5)
cbar = plt.colorbar() #ticks=[1e4,1e5,1e6,1e7])
cbar.set_label(r'$km\ s^{-1}$', size=10)
plt.savefig('Vgas_radial_'+test+'.pdf',dpi=100, bbox_inches='tight')

# Plotting theta velocity field  ------------------------------------
plt.figure(figsize=(10,6))
plt.pcolormesh(np.log10(rr/au),np.log10(zzr),abs(vtheta),norm=colors.LogNorm(vmin=1e-2,vmax=1),cmap='jet',shading='gouraud',rasterized=True)
#plt.pcolormesh(np.log10(rr/au),np.log10(zzr),rhod_biggr,cmap='gist_heat',norm=colors.LogNorm(vmin=3e-13,vmax=3e-10),shading='gouraud')
#plt.plot(np.log10(a_2/au),layer12,'k',lw=2,label=r'$\tau_{\rm 12CO}$=1')
#plt.plot(np.log10(a_2/au),layer13,'k--',lw=2,label=r'$\tau_{\rm 13CO}$=1')
plt.xlim(-1,2.4)
plt.ylim(-2,-0.25)
plt.xlabel(r'log$_{10}$(r/1 AU)', fontsize=15)
plt.ylabel(r'log$_{10}$(z/r)',fontsize=15)
#plt.legend(prop={'size':15},loc=1)
plt.tick_params(which='both',length=6,width=1.5)
cbar = plt.colorbar() #ticks=[1e4,1e5,1e6,1e7])
cbar.set_label(r'$km\ s^{-1}$', size=10)
plt.savefig('Vgas_theta_'+test+'.pdf',dpi=100, bbox_inches='tight')

# Plotting phi velocity field  ------------------------------------
plt.figure(figsize=(10,6))
plt.pcolormesh(np.log10(rr/au),np.log10(zzr),vphi,norm=colors.LogNorm(vmin=1e0,vmax=30),cmap='jet',shading='gouraud',rasterized=True)
#plt.pcolormesh(np.log10(rr/au),np.log10(zzr),rhod_biggr,cmap='gist_heat',norm=colors.LogNorm(vmin=3e-13,vmax=3e-10),shading='gouraud')
#plt.plot(np.log10(a_2/au),layer12,'k',lw=2,label=r'$\tau_{\rm 12CO}$=1')
#plt.plot(np.log10(a_2/au),layer13,'k--',lw=2,label=r'$\tau_{\rm 13CO}$=1')
plt.xlim(-1,2.4)
plt.ylim(-2,-0.25)
plt.xlabel(r'log$_{10}$(r/1 AU)', fontsize=15)
plt.ylabel(r'log$_{10}$(z/r)',fontsize=15)
#plt.legend(prop={'size':15},loc=1)
plt.tick_params(which='both',length=6,width=1.5)
cbar = plt.colorbar() #ticks=[1e4,1e5,1e6,1e7])
cbar.set_label(r'$km\ s^{-1}$', size=10)
plt.savefig('Vgas_phi_'+test+'.pdf',dpi=100, bbox_inches='tight')


#------------------------------------------------------------------------------------------------------------
# Plot for the T distribution  ------------------------------------
# Read the T file
test = args.file
fnameread      = 'Tdust_'+test+'.dat'
im_n, npop, temp = read_T_ascii(fnameread)
temp=temp.reshape((npop,ntheta,nr))
temp_smlgr=temp[0,:,:].T
temp_biggr=temp[1,:,:].T

# Set the initial T model by a passive heating disk  ------------------------------------
SB = 5.670374419e-5    # Stefan-Boltzmann constant [ erg cm-2 s-1 K-4]
mstar    = 0.8*ms # Stempels et al 2002
rstar    = 1.64*rs  # Herczeg et al. 2005
tstar    = 3950.#  Stempels et al. 2002
lstar    = 4*np.pi*rstar**2*SB*tstar**4
r0 = 10.*au
t0 = ( 0.05*lstar/(8*np.pi*r0**2*SB) )**0.25  # 30*(10**0.16)**0.25 #  Adopting the DSHARP values (10**0.16*ls), Passive heating model
pltt = -0.5
tdust = t0 * (rc/r0)**pltt

# Plot Tdust distribution in r,z plane ------------------------------------
plt.figure(figsize=(10,6))
plt.pcolormesh(np.log10(rr/au),np.log10(zzr),temp_biggr,cmap='jet',norm=colors.LogNorm(vmin=10,vmax=250),shading='gouraud',rasterized=True)
plt.xlim(-1,2.4)
plt.ylim(-2,-0.25)
plt.xlabel(r'log$_{10}$(r/1 AU)', fontsize=15)
plt.ylabel(r'log$_{10}$(z/r)',fontsize=15)
#plt.legend(prop={'size':15},loc=1)
plt.tick_params(which='both',length=6,width=1.5)
plt.colorbar(ticks=[10,20,30,40,50,60,70,80,90,100,200])
#plt.savefig('Tdust_rz_small.pdf',dpi=100, bbox_inches='tight')
plt.savefig('Tdust_rz_'+test+'.pdf',dpi=100,bbox_inches='tight')

# Plot Tgas distribution in r,z plane ------------------------------------
plt.figure(figsize=(10,6))
plt.pcolormesh(np.log10(rr/au),np.log10(zzr),temp_smlgr,cmap='jet',norm=colors.LogNorm(vmin=10,vmax=250),shading='gouraud',rasterized=True)
plt.xlim(-1,2.4)
plt.ylim(-2,-0.25)
plt.xlabel(r'log$_{10}$(r/1 AU)', fontsize=15)
plt.ylabel(r'log$_{10}$(z/r)',fontsize=15)
#plt.legend(prop={'size':15},loc=1)
plt.tick_params(which='both',length=6,width=1.5)
plt.colorbar(ticks=[10,20,30,40,50,60,70,80,90,100,200])
plt.savefig('Tgas_rz_'+test+'.pdf',dpi=100,bbox_inches='tight')
    
# Radial T profile at the midplane ------------------------------------
plt.figure(figsize=(10,6))
plt.plot(rr[:,-1]/au,temp_biggr[:,-1],'k',lw=2,label=r'Midplane T$_{\rm dust}$')
plt.plot(rr[:,-1]/au,temp_smlgr[:,-1],'k--',lw=2,label=r'Midplane T$_{\rm gas}$')
plt.plot(rc/au,tdust,'r',lw=2,label=r'Midplane T$_{\rm model}$')
plt.xlim(1,200)
plt.ylim(5,50)
plt.xlabel(r'log$_{10}$(r/1 AU)', fontsize=15)
plt.ylabel(r'T$_{dust}$',fontsize=15)
plt.legend(prop={'size':15},loc=0)
plt.tick_params(which='both',length=6,width=1.5)
plt.savefig('Tdust_radial_mid_'+test+'.pdf',dpi=100, bbox_inches='tight')


# Thermal broadening due to the gas temperature structure
#      Physical constants ----------------------------------
mu = 2.34
kb = 1.38e-16
NA = 6.02e23
mH = 1.0/NA
sigma=np.sqrt(2.*kb*temp_smlgr/mu/mH).reshape(nr,ntheta)*1e-5
# Plot Tdust distribution in r,z plane ------------------------------------
plt.figure(figsize=(10,6))
plt.pcolormesh(np.log10(rr/au),np.log10(zzr),sigma,cmap='jet',norm=colors.LogNorm(vmin=1e-1,vmax=1e0),shading='gouraud',rasterized=True)
plt.xlim(-1,2.4)
plt.ylim(-2,-0.25)
plt.xlabel(r'log$_{10}$(r/1 AU)', fontsize=15)
plt.ylabel(r'log$_{10}$(z/r)',fontsize=15)
#plt.legend(prop={'size':15},loc=1)
plt.tick_params(which='both',length=6,width=1.5)
plt.colorbar(ticks=[0.2,0.4,0.6,0.8])
#plt.savefig('Tdust_rz_small.pdf',dpi=100, bbox_inches='tight')
plt.savefig('Thermal_dV_'+test+'.pdf',dpi=100,bbox_inches='tight')

vturb = vphi*1e-3
#vturb = np.sqrt(2.*kb*temp_smlgr/mu/mH)*0.1
vr.max(), vtheta.min(),vturb.max(),sigma.max()

plt.figure(figsize=(10,6))
plt.pcolormesh(np.log10(rr/au),np.log10(zzr),abs(vtheta)/sigma,cmap='jet',norm=colors.LogNorm(vmin=1e-2,vmax=1e-1),shading='gouraud',rasterized=True)
plt.xlim(-1,2.4)
plt.ylim(-2,-0.25)
plt.xlabel(r'log$_{10}$(r/1 AU)', fontsize=15)
plt.ylabel(r'log$_{10}$(z/r)',fontsize=15)
#plt.legend(prop={'size':15},loc=1)
plt.tick_params(which='both',length=6,width=1.5)
plt.colorbar(ticks=[2e-2,4e-2,6e-2,8e02,1e-1])
#plt.savefig('Tdust_rz_small.pdf',dpi=100, bbox_inches='tight')
plt.savefig('Thermal_vs_wind_'+test+'.pdf',dpi=100,bbox_inches='tight')

plt.figure(figsize=(10,6))
plt.pcolormesh(np.log10(rr/au),np.log10(zzr),vturb/sigma,cmap='jet',norm=colors.LogNorm(vmin=5e-3,vmax=5e-2),shading='gouraud',rasterized=True)
plt.xlim(-1,2.4)
plt.ylim(-2,-0.25)
plt.xlabel(r'log$_{10}$(r/1 AU)', fontsize=15)
plt.ylabel(r'log$_{10}$(z/r)',fontsize=15)
#plt.legend(prop={'size':15},loc=1)
plt.tick_params(which='both',length=6,width=1.5)
plt.colorbar(ticks=[5e-3,75.e-3,1e-2,2.5e-2,5e-2])
#plt.savefig('Tdust_rz_small.pdf',dpi=100, bbox_inches='tight')
plt.savefig('Thermal_vs_turb_'+test+'.pdf',dpi=100,bbox_inches='tight')
