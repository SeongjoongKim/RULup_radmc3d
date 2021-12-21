import numpy as np
import matplotlib.colors as colors
from mpl_toolkits.mplot3d import axes3d
from matplotlib import pyplot as plt
from matplotlib import cm
from Model_setup_subroutines import *

# Some natural constants  ----------------------------------
au  = 1.49598e13     # Astronomical Unit       [cm]
pc  = 3.08572e18     # Parsec                  [cm]
ms  = 1.98892e33     # Solar mass              [g]
ts  = 5.78e3         # Solar temperature       [K]
ls  = 3.8525e33      # Solar luminosity        [erg/s]
rsun  = 6.96e10        # Solar radius            [cm]
GG  = 6.67408e-08    # Gravitational constant  [cm^3/g/s^2]
mp  = 1.6726e-24     # Mass of proton          [g]
SB = 5.670374419e-5    # Stefan-Boltzmann constant [ erg cm-2 s-1 K-4]
#      Physical constants ----------------------------------
mu = 2.34
kb = 1.38e-16
NA = 6.02e23
mH = 1.0/NA

mstar    = 0.63*ms # DSHARP
rstar    = 2.42*rsun  # DSHARP
tstar    = 4073.  # 3950.   #  4950 derived from 10**0.16 L_sun
lstar    = 4*np.pi*rstar**2*SB*tstar**4 #star_create(mstar,rstar,tstar,lam)

def Mass_loss(r_in,r_out,Cw,sigma0,m_star):   # Mloss(R) = int_0^R 2*(rho*vp)*2pi*r *dr
    a = 0.1*au; b = 1.0*au
    if (r_in < b) and (r_out <= b):
        integrated_in = (  a*r_in**(0.5) - (r_in**1.5)/3. )
        integrated_out = (  a*r_out**(0.5) - (r_out**1.5)/3. )
        M_loss = 8*np.pi*Cw*sigma0*(GG*m_star)**(0.5)/(1-(b/a))/a* ( integrated_out - integrated_in )
    elif (r_in > b) and (r_out > b):
        M_loss = -8.*np.pi*Cw*sigma0*(GG*m_star)**(0.5)*(r_out**(-0.5) - r_in**(-0.5))*au
    else:
        integrated_in = ( a*r_in**(0.5) - (r_in**1.5)/3. )
        integrated_out = (a*(1*au)**(0.5) - ((1*au)**1.5)/3. )
        M_loss = -8.*np.pi*Cw*sigma0*(GG*m_star)**(0.5)*(r_out**(-0.5) - (1.*au)**(-0.5))*au + 8*np.pi*Cw*sigma0*(GG*m_star)**(0.5)/(1-(b/a))/a* ( integrated_out - integrated_in )
    return M_loss
"""
def Mass_loss(r_in,r_out,Cw,sigma0,m_star):
    M_loss = -8.*np.pi*Cw*sigma0*(GG*m_star)**(0.5)*(r_out**(-0.5) - r_in**(-0.5))#*au
    return M_loss
"""
# Grid parameters  ----------------------------------
nr  = 100       # Radial grid（int）
ntheta = 100    # Elevation grid (z-direction, int)
nphi = 1        # Azimuthal grid 1 (y-direction, int)
rin = 1e-1 * au   # inner radius
rout = 300 * au # outer radius(cm)
thetaup = np.pi*0.5 - 0.7# Elevation limit。0<= theta <~ 50 deg
rc, thetac, phic = grid_set(nr,ntheta,nphi,rin,rout,thetaup)
# Make the grid ----------------------------------
qq       = np.meshgrid(rc,thetac,phic,indexing='ij')
rr       = qq[0]    # cm
tt       = qq[1]     # rad
zr       = np.pi/2.e0 - qq[1]     # rad
rs = rr*np.sin(tt)                # [cm] Cylindrical radius in 2D
zs = rr*np.cos(tt)                # [cm] Cylindrical z in 2D

#     Temperature parameters  ----------------------------------
r0 = 10.*au
t0 = ( 0.05*lstar/(8*np.pi*r0**2*SB) )**0.25    # 30*(10**0.16)**0.25 #  Adopting the DSHARP values (10**0.16*ls), Passive heating model
pltt = -0.5
tdust = t0 * (rs/r0)**pltt
cs       = np.sqrt(kb*tdust/(2.3*mp))   # Isothermal sound speed at midplane
omk      = np.sqrt(GG*mstar/rs**3)      # The Kepler angular frequency

# Disk parameters   ----------------------------------
hp       = cs/omk                      # The pressure scale height
hpr      = hp/rs                        # The dimensionless hp
#  Gas disk ----------------------------------
sigmag0  = 1.5e3
rg0          = 1.
rge0        = 50.
plsigg      =  -1.0

sigmag = np.zeros_like(rs)
a=0.1; b=1.0
for i in range(nr):
    for j in range(ntheta):
        if rs[i,j,0]>=1*au:
            sigmag[i,j,0]   = sigmag0*(rs[i,j,0]/rg0/au)**plsigg #  power-law function
        else:
            sigmag[i,j,0]   = sigmag0*(1.0-(rs[i,j,0]/a/au))*(1.0-(b/a))**(-1) #  power-law function

#sigmag   = sigmag0*(rs/rg0/au)**plsigg #  power-law function
sigmag[np.where(rs<0.1*au)] = 0.
beta_0 = 1.e4
Cw = 1.e-5
alpha_B = -2.0
P_mid = sigmag/np.sqrt(2.e0*np.pi)/hp *cs**2   # Equation (18) of Bai 2016, ApJ, 821, 80
B_mid = np.sqrt((8*np.pi*P_mid)/beta_0 )             # Equation (18) of Bai 2016, ApJ, 821, 80
z0 = 4.0*hp  #X*rs**(1.5+0.5*pltt)  # wind base
rhog = np.zeros_like(rs)   # [g/cm3] in Spherical (r,theta)
Bp = np.zeros_like(rs)
vz = np.zeros_like(rs)
# Calculate rho_gas, Bp, and vz: Bp and vz are values along cylindrical z axis.
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


flux = rhog*vz
sigmag_mid = np.zeros(len(rc))
M_loss = np.zeros(len(rc))
for i in range(len(rc)):
    M_loss[i] = Mass_loss(0.1*au,rc[i],Cw,sigmag0,mstar)/ms*(3600*24*365.24)

# Plotting mass flux field  ------------------------------------
plt.figure(figsize=(10,6))
plt.pcolormesh(np.log10(rr[:,:,0]/au),np.log10(zr[:,:,0]),flux[:,:,0],vmin=1e-13,vmax=1e-5, norm=colors.LogNorm(vmin=1e-16,vmax=1e-10),cmap='jet',shading='gouraud',rasterized=True)
#plt.pcolormesh(np.log10(rr[:,:,0]/au),np.log10(zr[:,:,0]),flux[:,:,0],vmin=1e-12,vmax=1e-5, norm=colors.LogNorm(vmin=1e-16,vmax=1e-10),cmap='jet',shading='gouraud')#,rasterized=True)
plt.plot(np.log10(rs[:,0,0]/au),np.log10(4*hp[:,0,0]/rs[:,0,0]),'k',lw=2,label=r'4 Hp')
plt.xlim(-1,2.4)
plt.ylim(-2,-0.25)
plt.xlabel(r'log$_{10}$(r/1 AU)', fontsize=15)
plt.ylabel(r'log$_{10}$(z/r)',fontsize=15)
plt.tick_params(which='both',length=6,width=1.5,direction='in',labelsize=15)
cbar = plt.colorbar() #ticks=[1e4,1e5,1e6,1e7])
cbar.set_label(r'$g\ cm^{-2}\ s^{-1}$', size=10)
plt.savefig('Wind_mass_flux_Cw{:4.0E}_beta{:4.0E}_rz.pdf'.format(Cw,beta_0),dpi=100, bbox_inches='tight')
#plt.show()
plt.clf()

# Plotting mass flux field  ------------------------------------
plt.figure(figsize=(10,6))
plt.pcolormesh(np.log10(rr[:,:,0]/au),np.log10(zr[:,:,0]),rhog[:,:,0],vmin=1e-16,vmax=1e-9, norm=colors.LogNorm(vmin=1e-16,vmax=1e-10),cmap='jet',shading='gouraud',rasterized=True)
#plt.pcolormesh(np.log10(rr[:,:,0]/au),np.log10(zr[:,:,0]),flux[:,:,0],vmin=1e-12,vmax=1e-5, norm=colors.LogNorm(vmin=1e-16,vmax=1e-10),cmap='jet',shading='gouraud')#,rasterized=True)
plt.plot(np.log10(rs[:,0,0]/au),np.log10(4*hp[:,0,0]/rs[:,0,0]),'k',lw=2,label=r'4 Hp')
plt.xlim(-1,2.4)
plt.ylim(-2,-0.25)
plt.xlabel(r'log$_{10}$(r/1 AU)', fontsize=15)
plt.ylabel(r'log$_{10}$(z/r)',fontsize=15)
plt.tick_params(which='both',length=6,width=1.5,direction='in',labelsize=15)
cbar = plt.colorbar() #ticks=[1e4,1e5,1e6,1e7])
cbar.set_label(r'$g\ cm^{-3}$', size=10)
plt.savefig('Wind_rhog_rz.pdf',dpi=100, bbox_inches='tight')
#plt.show()
plt.clf()


# Plotting mass loss profile  ------------------------------------
plt.figure(figsize=(10,6))
plt.plot(rc/au,M_loss,'k',lw=2,label=r'Mass loss rate')
plt.xlim(0.1,300)
#plt.ylim(1e-7,5e-5)
plt.xlabel(r'Radius [au]', fontsize=15)
plt.ylabel(r'$\dot{M}_{loss}$ [$M_{\odot}\ yr^{-1}$]',fontsize=15)
plt.tick_params(which='both',length=6,width=1.5,direction='in',labelsize=15)
plt.xscale('log')
plt.yscale('log')
#cbar = plt.colorbar() #ticks=[1e4,1e5,1e6,1e7])
#cbar.set_label(r'$g\ s^{-1}$', size=10)
plt.savefig('Wind_mass_loss_radial_Cw{:4.0E}_beta{:4.0E}.pdf'.format(Cw,beta_0),dpi=100, bbox_inches='tight')
#plt.show()
plt.clf()
