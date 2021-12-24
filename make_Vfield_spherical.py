import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors

# Some natural constants  ----------------------------------
au  = 1.49598e13     # Astronomical Unit       [cm]
pc  = 3.08572e18     # Parsec                  [cm]
ms  = 1.98892e33     # Solar mass              [g]
ts  = 5.78e3         # Solar temperature       [K]
ls  = 3.8525e33      # Solar luminosity        [erg/s]
rs  = 6.96e10        # Solar radius            [cm]
GG  = 6.67408e-08    # Gravitational constant  [cm^3/g/s^2]
mp  = 1.6726e-24     # Mass of proton          [g]
SB = 5.670374419e-5    # Stefan-Boltzmann constant [erg cm-2 s-1 K-4]

#      Physical constants ----------------------------------
mu = 2.34         # average molecular weight
kb = 1.38e-16     # boltzmann constant  [erg/K]
NA = 6.02e23      # Avogadro constant  [g^-1]
mH = 1.0/NA      # Hydrogen mass [g]

# Star parameters  ----------------------------------
mstar    = 0.63*ms # Stempels et al 2002
rstar    = 2.42*rs  # Herczeg et al. 2005
tstar    = 4073.#  Stempels et al. 2002
lstar    = 4*np.pi*rstar**2*SB*tstar**4
pstar    = np.array([0.,0.,0.])

# Grid parameters for Spherical ----------------------------------
nr  = 300       # Radial grid（int）
ntheta = 100    # Elevation grid (z-direction, int)
nphi = 1        # Azimuthal grid 1 (y-direction, int)
rin = 1e-1 * au   # inner radius
rout = 300 * au # outer radius(cm)
thetaup = np.pi*0.5 - 0.7# Elevation limit。0<= theta <~ 50 deg

# Make the coordinates  ----------------------------------
ri       = np.logspace(np.log10(rin),np.log10(rout),nr+1)      # [cm]
thetai   = np.linspace(thetaup,0.5e0*np.pi,ntheta+1)        # [rad]
phii     = np.linspace(0.e0,np.pi*2.e0,nphi+1)                    # [rad]
rc       = 0.5 * ( ri[0:nr] + ri[1:nr+1] )
thetac   = 0.5 * ( thetai[0:ntheta] + thetai[1:ntheta+1] )
phic     = 0.5 * ( phii[0:nphi] + phii[1:nphi+1] )
# Make the grid ----------------------------------
qq       = np.meshgrid(rc,thetac,phic,indexing='ij')
rr       = qq[0]    # cm
tt       = qq[1]     # rad  Spherical theta
zr       = np.pi/2.e0 - qq[1]     # rad  Cylindrical z in angle
rs = rr*np.sin(tt)                # [cm] Cylindrical radius in 2D
zs = rs*np.cos(tt)                # [cm] Cylindrical z in 2D

#     Temperature parameters  ----------------------------------
r0 = 10.*au
t0 = ( 0.05*lstar/(8*np.pi*r0**2*SB) )**0.25    # 30*(10**0.16)**0.25 # Adopting the DSHARP values
pltt = -0.5
tdust = t0 * (rs/r0)**pltt   # 2D T_mid distribution in (r,theta)
cs       = np.sqrt(kb*tdust/(2.3*mp))   # [cm/s]Isothermal sound speed
omk      = np.sqrt(GG*mstar/rs**3)      # [s^-1] The Kepler angular frequency
hp       = cs/omk                     # [cm] The pressure scale height
hpr      = hp/rs                        # The dimensionless hp
#zb   = 0.15*rs
#cv   = 0.1

# Disk parameters   ----------------------------------
#  Gas disk ----------------------------------
sigmag0  = 1.5e3 #sigmad0*1e2*(100./rd0)**plsig
rg0          = 1.
rge0        = 50.
plsigg      =  -1.0
#  Dust disk ----------------------------------
sigmad0 = 1.5e2      # dust surface density at rd0 au     [g/cm2]
plsig    = -1.0            # Powerlaw of the surface density
rd0     = 1              # Sigma_dust,0 at 5 au
rde0   = 50.           # exponential tail of dust profile
#plh      = 1.25               # Powerlaw of flaring

# Make the density model ----------------------------------
sigmag   = sigmag0*(rs/rg0/au)**plsigg #  power-law function
beta_0 = 1.e4
Cw = 1.0e-4
alpha_B = -1.0
P_mid = sigmag/np.sqrt(2.e0*np.pi)/hp *cs**2   # Equation (18) of Bai 2016, ApJ, 821, 80
B_mid = np.sqrt((8*np.pi*P_mid)/beta_0 )             # Equation (18) of Bai 2016, ApJ, 821, 80
z0 = 4.0*hp  #X*rs**(1.5+0.5*pltt)  # wind base
rhog = np.zeros_like(rs)   # [g/cm3] in Spherical (r,theta)
Bp = np.zeros_like(rs)
vz = np.zeros_like(rs)
a=0.1; b=1.0             # Set the disk inner edge (0.1 au) and tapering radius (1 au)
for i in range(0,nr):
    for j in range(0,ntheta):
        if (zs[i,j,0]<=z0[i,j,0]):
            rhog[i,j,0] = ( sigmag[i,j,0] / (np.sqrt(2.e0*np.pi)*hp[i,j,0]) ) * np.exp(-zs[i,j,0]**2/hp[i,j,0]**2/2.e0)  #[g/cm3]
            Bp[i,j,0] = B_mid[i,j,0]
            vz[i,j,0] = Cw*sigmag[i,j,0]*omk[i,j,0]/rhog[i,j,0]   # Suzuki et al. 2010, ApJ, 718, 1289
        else:
            R0 = z0[i,j,0]-zs[i,j,0]+rs[i,j,0]
            if R0 < 0.1*au:
                rhog[i,j,0] = 0.
                Bp[i,j,0] = 0.
                vz[i,j,0] = 0.
            else:
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
                rhog[i,j,0] = rhog_R0 * np.exp(-8.0) * np.exp(-(cs_R0/omk_R0)**(-0.6) * np.sqrt((zs[i,j,0]-(4*hp_R0))/R0 ) )
                Bp[i,j,0] = B_mid_R0* (rs[i,j,0]/R0)**alpha_B #*np.cos(45.*np.pi/180.)  # Bai 2016, ApJ, 818,152
                vz[i,j,0] = Cw*sigmag_R0*omk_R0*Bp[i,j,0]/B_mid_R0/rhog[i,j,0] #*np.cos(45.*np.pi/180.)


#Bp[np.isnan(Bp)] = 0
#rhog[np.isnan(rhog)] = 0
# Make the velocity model ----------------------------------
# Set up cylindrical coordinate data
#vr       = np.zeros_like(rr)
#vtheta   = np.zeros_like(rr)
vphi = np.sqrt(GG*mstar/rr)
#vz = Cw*sigmag*omk[:,None,None]/rhog*Bp
#vz[np.isnan(vz)] = 0
#vphi = np.sqrt(vkep**2 - vz**2)
#vr = np.zeros_like(vz)
#vtheta = vz*np.cos((tt))
#vturb    = 0.1*vphi
vturb    = 0.001*vphi

"""
with open('gas_density.inp','w+') as f:
    f.write('1\n')                       # Format number
    f.write('%d\n'%(nr*nz*ntheta))     # Nr of cells
    f.write('1\n')                       # Nr of dust species
    data = rhog.ravel(order='F')         # Create a 1-D view, fortran-style indexing
    data.tofile(f, sep='\n', format="%13.6e")
    f.write('\n')

# Write the gas velocity field   ----------------------------------
with open('gas_velocity.inp','w+') as f:
    f.write('1\n')                       # Format number
    f.write('%d\n'%(nr*nz*ntheta))     # Nr of cells
    for iphi in range(ntheta):
        for itheta in range(nz):
            for ir in range(nr):
                f.write('%13.6e %13.6e %13.6e\n'%(vr[ir,itheta,iphi],vtheta[ir,itheta,iphi],vphi[ir,itheta,iphi]))

with open('microturbulence.inp','w+') as f:
    f.write('1\n')                       # Format number
    f.write('%d\n'%(nr*nz*ntheta))     # Nr of cells
    data = vturb.ravel(order='F')        # Create a 1-D view, fortran-style indexing
    data.tofile(f, sep='\n', format="%13.6e")
    f.write('\n')
"""
"""
ir1 = 10; ir2 = 30; ir3 = 50; ir4 = 70
fig=plt.figure(num=None, figsize=(8,6), dpi=400, facecolor='w', edgecolor='k')
plt.subplot(111)
plt.plot(zz[ir1,:,0]/rr[ir1,:,0],rhog[ir1,:,0],'k',label=str(rc[ir1]/au)+' au')
plt.plot(zz[ir2,:,0]/rr[ir2,:,0],rhog[ir2,:,0],'r',label=str(rc[ir2]/au)+' au')
plt.plot(zz[ir3,:,0]/rr[ir3,:,0],rhog[ir3,:,0],'b',label=str(rc[ir3]/au)+' au')
plt.plot(zz[ir4,:,0]/rr[ir4,:,0],rhog[ir4,:,0],'g',label=str(rc[ir4]/au)+' au')
plt.xlabel('z/r [rad]')
plt.ylabel(r'$\rho$ [g/cm$^3$]')
#plt.xlim([1e-2,1e0])
#plt.ylim([1e-15,1e-10])
plt.yscale('log')
plt.xscale('log')
plt.legend(prop={'size':12},loc=0)
plt.tight_layout()
plt.savefig('rho_z_direc_test.pdf')

fig=plt.figure(num=None, figsize=(8,6), dpi=400, facecolor='w', edgecolor='k')
plt.subplot(111)
plt.plot(zz[ir1,:,0]/rr[ir1,:,0],vz[ir1,:,0],'k',label=str(rc[ir1]/au)+' au')
plt.plot(zz[ir2,:,0]/rr[ir2,:,0],vz[ir2,:,0],'r',label=str(rc[ir2]/au)+' au')
plt.plot(zz[ir3,:,0]/rr[ir3,:,0],vz[ir3,:,0],'b',label=str(rc[ir3]/au)+' au')
plt.plot(zz[ir4,:,0]/rr[ir4,:,0],vz[ir4,:,0],'g',label=str(rc[ir4]/au)+' au')
plt.xlabel('z/r [rad]')
plt.ylabel(r'V$_{z}$ [cm/s$^{-1}$]')
#plt.xlim([1e-2,1e0])
#plt.ylim([1e-15,1e-10])
plt.yscale('log')
plt.xscale('log')
plt.legend(prop={'size':12},loc=0)
plt.tight_layout()
plt.savefig('Vel_z_direc_test.pdf')


fig=plt.figure(num=None, figsize=(8,6), dpi=400, facecolor='w', edgecolor='k')
plt.subplot(111)
plt.plot(rc/au,zb/au,'r',label='Zb = 0.15R0')
plt.plot(rc/au,4*hp/au,'b',label='4 Hp')
plt.xlabel('radius [au]')
plt.ylabel(r'z [au]')
#plt.xlim([1e-2,1e0])
#plt.ylim([1e-15,1e-10])
plt.yscale('log')
plt.xscale('log')
plt.legend(prop={'size':12},loc=0)
plt.tight_layout()
plt.savefig('zb_hp_test.pdf')
"""

plt.figure(figsize=(10,6))
plt.pcolormesh(np.log10(rr[:,:,0]/au),np.log10(zr[:,:,0]),rhog[:,:,0],norm=colors.LogNorm(vmin=1e-16,vmax=1e-9),cmap='jet',shading='gouraud',rasterized=True)
#plt.pcolormesh(np.log10(rr/au),np.log10(zzr),rhod_biggr,cmap='gist_heat',norm=colors.LogNorm(vmin=3e-13,vmax=3e-10),shading='gouraud')
plt.plot(np.log10(rs[:,0,0]/au),np.log10(4*hp[:,0,0]/rs[:,0,0]),'k',lw=2,label=r'4 Hp')
#plt.plot(np.log10(rr[:,0,0]/au),np.log10(zb/rr[:,0,0]),'k--',lw=2,label=r'Zb')
plt.xlim(-1,2.4)
plt.ylim(-2,-0.2)
plt.xlabel(r'log$_{10}$(r$_{sph}$/1 AU)', fontsize=15)
plt.ylabel(r'log$_{10}$($\theta_{sph}$)',fontsize=15)
plt.legend(prop={'size':15},loc=0)
plt.tick_params(which='both',length=6,width=1.5)
plt.colorbar() #ticks=[1e-15,1e-14,1e-13,1e-12])
plt.savefig('gas_dens_spherical_rz.pdf',dpi=100, bbox_inches='tight')
plt.clf()


plt.figure(figsize=(10,6))
plt.pcolormesh(np.log10(rr[:,:,0]/au),np.log10(zr[:,:,0]),vz[:,:,0]*1e-5,norm=colors.LogNorm(vmin=1e-2,vmax=1e1),cmap='jet',shading='gouraud',rasterized=True)
#plt.pcolormesh(np.log10(rr/au),np.log10(zzr),rhod_biggr,cmap='gist_heat',norm=colors.LogNorm(vmin=3e-13,vmax=3e-10),shading='gouraud')
plt.plot(np.log10(rs[:,0,0]/au),np.log10(4*hp[:,0,0]/rs[:,0,0]),'k',lw=2,label=r'4 Hp')
#plt.plot(np.log10(rr[:,0,0]/au),np.log10(zb/rr[:,0,0]),'k--',lw=2,label=r'Zb')
plt.xlim(-1.0,2.4)
plt.ylim(-2.0,-0.2)
plt.xlabel(r'log$_{10}$(r$_{sph}$/1 AU)', fontsize=15)
plt.ylabel(r'log$_{10}$($\theta_{sph}$)',fontsize=15)
plt.legend(prop={'size':15},loc=0)
plt.tick_params(which='both',length=6,width=1.5)
cbar = plt.colorbar() #ticks=[1e4,1e5,1e6,1e7])
cbar.set_label(r'$km\ s^{-1}$', size=10)
plt.savefig('Vz_spherical_B-1_rz.pdf',dpi=100, bbox_inches='tight')
plt.clf()

plt.figure(figsize=(10,6))
plt.pcolormesh(np.log10(rr[:,:,0]/au),np.log10(zr[:,:,0]),Bp[:,:,0],norm=colors.LogNorm(vmin=1e-3,vmax=1e0),cmap='jet',shading='gouraud',rasterized=True)
#plt.pcolormesh(np.log10(rr/au),np.log10(zzr),rhod_biggr,cmap='gist_heat',norm=colors.LogNorm(vmin=3e-13,vmax=3e-10),shading='gouraud')
plt.plot(np.log10(rs[:,0,0]/au),np.log10(4*hp[:,0,0]/rs[:,0,0]),'k',lw=2,label=r'4 Hp')
#plt.plot(np.log10(rr[:,0,0]/au),np.log10(zb/rr[:,0,0]),'k--',lw=2,label=r'Zb')
plt.xlim(-1,2.4)
plt.ylim(-2,-0.2)
plt.xlabel(r'log$_{10}$(r$_{sph}$/1 AU)', fontsize=15)
plt.ylabel(r'log$_{10}$($\theta_{sph}$)',fontsize=15)
plt.legend(prop={'size':15},loc=0)
plt.tick_params(which='both',length=6,width=1.5)
plt.colorbar() #ticks=[1e4,1e5,1e6,1e7])
plt.savefig('Bz_spherical_B-1_rz.pdf',dpi=100, bbox_inches='tight')
plt.clf()

"""

# Testing the data along the field line
alpha_B = -2.0
RR = [1., 5., 10., 50., 100.]
color = ['k','r','g','b','m']
fig=plt.figure(num=None, figsize=(8,6), dpi=100, facecolor='w', edgecolor='k')
ax1 =plt.subplot()
ax2 = ax1.twinx()
for i in range(len(RR)):
    R0 = RR[i]*au
    sigmag_R0 = sigmag0*(R0/rg0/au)**plsigg
    t_R0 = t0 * (R0/r0)**pltt
    cs_R0 = np.sqrt(kb*t_R0/(2.3*mp))
    omk_R0 = np.sqrt(GG*mstar/R0**3)
    hp_R0 = cs_R0 / omk_R0
    Z0 = 4.*hp_R0
    rhog_R0 = sigmag_R0 / np.sqrt(2.e0*np.pi)/hp_R0
    P_mid_R0 = rhog_R0*cs_R0**2
    B_mid_R0 = np.sqrt((8*np.pi*P_mid_R0)/beta_0)
    vp_R0 = Cw*sigmag_R0*omk_R0/rhog_R0
    r = np.arange(100)*au+R0
    z = np.arange(100)*au+Z0
    omk = np.sqrt(GG*mstar/r**3)
    rhog = rhog_R0 * np.exp(-8.0) * np.exp(-(2.*cs_R0/omk_R0)**(-0.6) * np.sqrt((z-Z0)/R0 ) )
    Bp = B_mid_R0* (r/R0)**alpha_B
    vp = Cw*sigmag_R0*omk_R0*(Bp/B_mid_R0)/rhog
    ax1.plot(r/au,Bp,color[i],label='Bp along field line at R0 = {:4.1f} au'.format(R0/au) )
    ax2.plot(r/au,vp*1e-5,color[i],ls='--')
ax1.set_xlabel('$(dr^2 + dz^2)^{0.5}$ [au]')
ax1.set_ylabel('Bp')
ax2.set_ylabel('Vp [km/s]')
#ax1.set_xlim([1e-2,1e0])
#ax1.set_ylim([1e-15,1e-10])
ax1.set_yscale('log')
#ax1.set_xscale('log')
ax1.legend(prop={'size':12},loc=0)
plt.tight_layout()
plt.savefig('Bp_test.pdf')
plt.clf()

"""
