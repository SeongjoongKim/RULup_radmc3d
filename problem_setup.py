import numpy as np
from Model_setup_subroutines import *
import argparse

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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-calmode', default='I', type=str, help='Calculation mode: T (temperature) or I (image)')
    parser.add_argument('-wind', default='F', type=str, help='If you want to include wind model, set this argument T. default is F (no wind).')
    parser.add_argument('-settle', default=0.1, type=float, help='dust settling degree (Hd/Hg). Please set between 0.1~1.0')
    args = parser.parse_args()
    print(args)
    
    # Write the wavelength_micron.inp file  ----------------------------------
    lam1     = 0.1e0
    lam2     = 7.0e0
    lam3     = 25.e0
    lam4     = 1.0e5
    n12      = 20
    n23      = 100
    n34      = 60
    lam, nlam = lambda_create(lam1,lam2,lam3,lam4,n12,n23,n34)

    # Star parameters  ----------------------------------
    #mstar    = 0.8*ms # Stempels et al 2002
    #rstar    = 1.64*rs  # Herczeg et al. 2005
    #tstar    = 3950  # 3950.   #  4950 derived from 10**0.16 L_sun
    mstar    = 0.63*ms # DSHARP
    rstar    = 2.42*rsun  # DSHARP
    tstar    = 4073.  # 3950.   #  4950 derived from 10**0.16 L_sun
    lstar    = star_create(mstar,rstar,tstar,lam)

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
    settfact = args.settle               # Settling factor of the big grains
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
    sigmag = np.zeros_like(rs)
    sigmad = np.zeros_like(rs)
    a=0.1; b=1.0             # Set the disk inner edge (0.1 au) and tapering radius (1 au)
    for i in range(nr):
        for j in range(ntheta):
            if rs[i,j,0]>=1*au:
                sigmag[i,j,0]   = sigmag0*(rs[i,j,0]/rg0/au)**plsigg #  power-law function
                sigmad[i,j,0]   = sigmad0*(rs[i,j,0]/rd0/au)**plsig*np.exp(-(rs[i,j,0]/rde0/au)**2.0) #  power-law function
            else:
                sigmag[i,j,0]   = sigmag0*(1.0-(rs[i,j,0]/a/au))*(1.0-(b/a))**(-1) #  power-law function
                sigmad[i,j,0]   = sigmad0*(1.0-(rs[i,j,0]/a/au))*(1.0-(b/a))**(-1) #  power-law function
    sigmag[np.where(rs<0.1*au)] = 0.
    if args.wind == 'F':
        # Gas density
        #sigmag   = sigmag0*(rs/rg0/au)**plsigg #  power-law function
        rhog = rho_normal(rs,zs,sigmag,hp)
        # Dust density
        #sigmad  = sigmad0*(rs/rd0/au)**plsig*np.exp(-(rs/rde0/au)**2.0)
        rhod = rho_normal(rs,zs,sigmad,hp_biggr)
        rhos = rhog*1.0e-3
    elif args.wind == 'T':
        # Gas density
        #sigmag   = sigmag0*(rs/rg0/au)**plsigg #  power-law function
        beta_0 = 1.e4
        Cw = 1.0e-5
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
        # Dust density
        #sigmad  = sigmad0*(rs/rd0/au)**plsig*np.exp(-(rs/rde0/au)**2.0)
        rhod = rho_normal(rs,zs,sigmad,hp_biggr)
        rhos = rhog*1.0e-3
    else:
        raise ValueError("Unknown wind model")

    # Molecular fractional abundance
    molecule_abun = fractional_abundance(rs,zs,'12CO')
    rhoCO     = rhog * molecule_abun       #[g/cm3]
    molecule_abun = fractional_abundance(rs,zs,'13CO')
    rho13CO     = rhog * molecule_abun       #[g/cm3]
    molecule_abun = fractional_abundance(rs,zs,'C18O')
    rhoC18O     = rhog * molecule_abun       #[g/cm3]
    molecule_abun = fractional_abundance(rs,zs,'CN')
    rhoCN     = rhog * molecule_abun       #[g/cm3]

    # Writing the numberdensity input file
    write_ndens('co',rhoCO,28.0,nr,ntheta,nphi)
    write_ndens('13co',rho13CO,29.0,nr,ntheta,nphi)
    write_ndens('c18o',rhoC18O,30.0,nr,ntheta,nphi)
    write_ndens('cn',rhoCN,26.0,nr,ntheta,nphi)
    write_line_all(LTE=True)
    # H2 numberdensity files for collisional excitation
    write_ndens('h2',rhog,2.34,nr,ntheta,nphi)
    write_ndens('o-h2',0.75*rhog,2.34,nr,ntheta,nphi)
    write_ndens('p-h2',0.25*rhog,2.34,nr,ntheta,nphi)

    if args.calmode == 'I':
        # Writing the density input file
        write_density(rhod,rhog,nr,ntheta,nphi)
        write_opacity(1)
    elif args.calmode == 'T':
        write_density_2grain(rhod,rhos,rhog,nr,ntheta,nphi)
        write_opacity(2)
    else:
        raise ValueError("Unknown calmode")

    # Make the velocity model ----------------------------------
    vr       = np.zeros_like(rr)
    vtheta   = np.zeros_like(rr)
    vphi = np.sqrt(GG*mstar/rr)
    vturb    = 1e-3*vphi
    if args.wind == 'T':
        vr = vz*np.cos(tt)
        vtheta = -vz*np.sin(tt)
    write_velocity(vr,vtheta,vphi,vturb,nr,ntheta,nphi)

    # Monte Carlo parameters ----------------------------------
    if args.calmode == 'T':
        nphot    = 3000000   # for fiducial_wind, 1e6 due to the limit of cal. time
    elif args.calmode == 'I':
        nphot    = 1000000
    write_radmc_input(nphot,2,LTE=True)


if __name__ == '__main__':
    main()
