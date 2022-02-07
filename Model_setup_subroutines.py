import numpy as np
import scipy.interpolate as inter

# Some natural constants  ----------------------------------
au  = 1.49598e13     # Astronomical Unit       [cm]
pc  = 3.08572e18     # Parsec                  [cm]
ms  = 1.98892e33     # Solar mass              [g]
ts  = 5.78e3         # Solar temperature       [K]
ls  = 3.8525e33      # Solar luminosity        [erg/s]
rs  = 6.96e10        # Solar radius            [cm]
GG  = 6.67408e-08    # Gravitational constant  [cm^3/g/s^2]
mp  = 1.6726e-24     # Mass of proton          [g]
SB = 5.670374419e-5    # Stefan-Boltzmann constant [ erg cm-2 s-1 K-4]
h = 6.62607004e-27       # Planck function [ cm2 g s-1]
#      Physical constants ----------------------------------
mu = 2.34
kb = 1.38e-16
NA = 6.02e23
mH = 1.0/NA

def fit(x,a0,b0,c0):
    return a0*(1 + (x/b0)**c0 )
    
def Gaussian_func(x, a0, b0, c0):
    return a0*np.exp(-0.5*(x-b0)**2.0/c0**2.0)
    
def Power_law(x, a0, b0, c0):
    return a0* (x/b0)**c0

def lambda_create(lam1,lam2,lam3,lam4,n12,n23,n34):
    lam12    = np.logspace(np.log10(lam1),np.log10(lam2),n12,endpoint=False)
    lam23    = np.logspace(np.log10(lam2),np.log10(lam3),n23,endpoint=False)
    lam34    = np.logspace(np.log10(lam3),np.log10(lam4),n34,endpoint=True)
    lam      = np.concatenate([lam12,lam23,lam34])
    nlam     = lam.size
    # Write the wavelength file  ----------------------------------
    with open('wavelength_micron.inp','w+') as f:
        f.write('%d\n'%(nlam))
        for value in lam:
            f.write('%21.14e\n'%(value))
    return lam, nlam
    

def star_create(mstar,rstar,tstar,lam):
    lstar    = 4*np.pi*rstar**2*SB*tstar**4
    pstar    = np.array([0.,0.,0.])
    # Write the stars.inp file  ----------------------------------
    nlam = lam.size
    with open('stars.inp','w+') as f:
        f.write('2\n')
        f.write('1 %d\n\n'%(nlam))
        f.write('%21.14e %21.14e %21.14e %21.14e %21.14e\n\n'%(rstar,mstar,pstar[0],pstar[1],pstar[2]))
        for value in lam:
            f.write('%21.14e\n'%(value))
        f.write('\n%21.14e\n'%(-tstar))
    return lstar


def grid_set(nr,ntheta,nphi,rin,rout,thetaup):
    # Make the coordinates  ----------------------------------
    ri       = np.logspace(np.log10(rin),np.log10(rout),nr+1)      # [cm]
    thetai   = np.linspace(thetaup,0.5e0*np.pi,ntheta+1)        # [rad]
    phii     = np.linspace(0.e0,np.pi*2.e0,nphi+1)                    # [rad]
    rc       = 0.5 * ( ri[0:nr] + ri[1:nr+1] )
    thetac   = 0.5 * ( thetai[0:ntheta] + thetai[1:ntheta+1] )
    phic     = 0.5 * ( phii[0:nphi] + phii[1:nphi+1] )
    # Write the grid file   ----------------------------------
    with open('amr_grid.inp','w+') as f:
        f.write('1\n')                       # iformat
        f.write('0\n')                       # AMR grid style  (0=regular grid, no AMR)
        f.write('100\n')                     # Coordinate system: <100: Cartesian | 100<= <200:  spherical  | 200<= 300: Cylindrycal
        f.write('0\n')                       # gridinfo
        f.write('1 1 0\n')                   # Include r,theta, phi in coordinates                    < ---------------------------------- Check for making 1D, 2D, or 3D
        f.write('%d %d %d\n'%(nr,ntheta,nphi))  # Size of grid
        for value in ri:
            f.write('%13.6e\n'%(value))      # X coordinates (cell walls)
        for value in thetai:
            f.write('%13.6e\n'%(value))      # Y coordinates (cell walls)
        for value in phii:
            f.write('%13.6e\n'%(value))      # Z coordinates (cell walls)
    return rc, thetac, phic


def rho_normal(rs,zs,sig,hp):
    rho     = ( sig / (np.sqrt(2.e0*np.pi)*hp) ) * np.exp(-(zs**2/hp**2)/2.e0)#*np.exp(-(rr/2.25e2/au)**10)
    return rho
    

def fractional_abundance(rs,zs,species):
    if species == '12CO':
        fa = 1e-4
        return fa
    if species == '13CO':
        #fa = 1e-4/67.*np.exp(-(rs/50./au)**(2.))
        fa = 1e-4/67.*np.exp(-(rs/50./au))
        return fa
    if species == 'C18O':
        #fa = 1e-4/67./6.*np.exp(-(rs/50./au)**(2.))
        fa = 1e-4/67./6.*np.exp(-(rs/50./au))
        return fa
    if species == 'CN':
        fita = 0.35; fitb = 290.; fitc = 1.4
        #fita = 0.25; fitb = 100.; fitc = 1.0
        zcn = rs * fita* (1+ (rs/fitb/au )**fitc ) # (0.1 + 0.07*(rr/9./au)**0.6)
        iup = np.where(zs>=zcn)
        idown = np.where(zs<zcn)
        H_CN = np.zeros_like(rs)
        H_CN[iup] = 0.15*zcn[iup]
        H_CN[idown] = 0.125*zcn[idown]
        fa = np.zeros_like(rs)
        fa[iup] = 1e-6*np.exp(-0.5*((zs[iup]-zcn[iup])/H_CN[iup])**2.0)
        fa[idown] = 1e-6*np.exp(-0.5*((zcn[idown]-zs[idown])/H_CN[idown])**2.0)
        return fa


def write_density(rhod,rhog,nr,ntheta,nphi):
    # Write the density file  ----------------------------------
    with open('dust_density.inp','w+') as f:
        f.write('1\n')                       # Format number
        f.write('%d\n'%(nr*ntheta*nphi))     # Nr of cells
        f.write('1\n')                       # Nr of dust species
        data = rhod.ravel(order='F')         # Create a 1-D view, fortran-style indexing
        data.tofile(f, sep='\n', format="%13.6e")
        f.write('\n')
    with open('gas_density.inp','w+') as f:
        f.write('1\n')                       # Format number
        f.write('%d\n'%(nr*ntheta*nphi))     # Nr of cells
        f.write('1\n')                       # Nr of dust species
        data = rhog.ravel(order='F')         # Create a 1-D view, fortran-style indexing
        data.tofile(f, sep='\n', format="%13.6e")
        f.write('\n')
    return


def write_density_2grain(rhod,rhos,rhog,nr,ntheta,nphi):
    # Write the density file  ----------------------------------
    with open('dust_density.inp','w+') as f:
        f.write('1\n')                       # Format number
        f.write('%d\n'%(nr*ntheta*nphi))     # Nr of cells
        f.write('2\n')                       # Nr of dust species
        data = rhos.ravel(order='F')         # Create a 1-D view, fortran-style indexing
        data.tofile(f, sep='\n', format="%13.6e")
        f.write('\n')
        data = rhod.ravel(order='F')         # Create a 1-D view, fortran-style indexing
        data.tofile(f, sep='\n', format="%13.6e")
        f.write('\n')
    with open('gas_density.inp','w+') as f:
        f.write('1\n')                       # Format number
        f.write('%d\n'%(nr*ntheta*nphi))     # Nr of cells
        f.write('1\n')                       # Nr of dust species
        data = rhog.ravel(order='F')         # Create a 1-D view, fortran-style indexing
        data.tofile(f, sep='\n', format="%13.6e")
        f.write('\n')
    return


def write_ndens(mole,rho,mu,nr,ntheta,nphi):
    # Write the molecule number density file.   ----------------------------------
    n    = rho/(mu*mp)  #*factco
    with open('numberdens_'+mole+'.inp','w+') as f:
        f.write('1\n')                       # Format number
        f.write('%d\n'%(nr*ntheta*nphi))     # Nr of cells
        data = n.ravel(order='F')          # Create a 1-D view, fortran-style indexing
        data.tofile(f, sep='\n', format="%13.6e")
        f.write('\n')
    return

def write_line(mole,LTE=True):
    if LTE==False:
        # Write the lines.inp control file   ----------------------------------
        with open('lines.inp','w') as f:
            f.write('2\n')
            f.write('1\n')
            f.write(mole+'    leiden    0    0    2\n')
            f.write('p-h2\n')             #    non-LTE optically thin line tracing
            f.write('o-h2\n')             #    non-LTE optically thin line tracing
        return
    else:
        # Write the lines.inp control file   ----------------------------------
        with open('lines.inp','w') as f:
            f.write('2\n')
            f.write('1\n')
            f.write(mole+'    leiden    0    0    0\n')
        return
        

def write_line_all(LTE=True):
    if LTE==False:
        # Write the lines.inp control file   ----------------------------------
        with open('lines.inp','w') as f:
            f.write('2\n')
            f.write('4\n')
            f.write('co    leiden    0    0    2\n')
            f.write('p-h2\n')             #    non-LTE optically thin line tracing
            f.write('o-h2\n')             #    non-LTE optically thin line tracing
            f.write('13co    leiden    0    0    2\n')
            f.write('p-h2\n')             #    non-LTE optically thin line tracing
            f.write('o-h2\n')             #    non-LTE optically thin line tracing
            f.write('c18o    leiden    0    0    2\n')
            f.write('p-h2\n')             #    non-LTE optically thin line tracing
            f.write('o-h2\n')             #    non-LTE optically thin line tracing
            f.write('cn    leiden    0    0    2\n')
            f.write('p-h2\n')             #    non-LTE optically thin line tracing
            f.write('o-h2\n')             #    non-LTE optically thin line tracing
        return
    else:
        # Write the lines.inp control file   ----------------------------------
        with open('lines.inp','w') as f:
            f.write('2\n')
            f.write('4\n')
            f.write('co    leiden    0    0    0\n')
            f.write('13co    leiden    0    0    0\n')
            f.write('c18o    leiden    0    0    0\n')
            f.write('cn    leiden    0    0    0\n')
        return


def write_opacity(ngrain,scat):
    if ngrain == 1:
        # Dust opacity control file  ----------------------------------
        with open('dustopac.inp','w+') as f:
            f.write('2               Format number of this file\n')
            f.write('1               Nr of dust species\n')
            f.write('============================================================================\n')
            if scat == 0: f.write('1               Way in which this dust species is read\n')    # 1 = kappa_*.inp  10 = kapscatmat_*.inp
            if scat == 1: f.write('10               Way in which this dust species is read\n')    # 1 = kappa_*.inp  10 = kapscatmat_*.inp
            f.write('0               0=Thermal grain\n')
            if scat == 0: f.write('amax1mm_kaponly        Extension of name of dustkappa_***.inp file\n')
            if scat == 1: f.write('DSHARP_amax10000.0um        Extension of name of dustkapscatmat_***.inp file\n')
            #f.write('dsharp_wav8.70e+02mic_amax1.50e+02mic        Extension of name of dustkappa_***.inp file\n')
            #f.write('dsharp1000um        Extension of name of dustkapscatmat_***.inp file\n')
            f.write('----------------------------------------------------------------------------\n')
        return
    if ngrain == 2:
        # Dust opacity control file  ----------------------------------
        with open('dustopac.inp','w+') as f:
            f.write('2               Format number of this file\n')
            f.write('2               Nr of dust species\n')
            f.write('============================================================================\n')
            if scat == 0: f.write('1               Way in which this dust species is read\n')    # 1 = kappa_*.inp  10 = kapscatmat_*.inp
            if scat == 1: f.write('10               Way in which this dust species is read\n')    # 1 = kappa_*.inp  10 = kapscatmat_*.inp
            f.write('0               0=Thermal grain\n')
            if scat == 0: f.write('amax10um_kaponly        Extension of name of dustkappa_***.inp file\n')
            if scat == 1: f.write('DSHARP_amax10.0um        Extension of name of dustkapscatmat_***.inp file\n')
            #f.write('dsharp10um        Extension of name of dustkapscatmat_***.inp file\n')
            f.write('----------------------------------------------------------------------------\n')
            if scat == 0: f.write('1               Way in which this dust species is read\n')    # 1 = kappa_*.inp  10 = kapscatmat_*.inp
            if scat == 1: f.write('10               Way in which this dust species is read\n')    # 1 = kappa_*.inp  10 = kapscatmat_*.inp
            f.write('0               0=Thermal grain\n')
            if scat == 0: f.write('amax1mm_kaponly        Extension of name of dustkappa_***.inp file\n')
            if scat == 1: f.write('DSHARP_amax10000.0um        Extension of name of dustkapscatmat_***.inp file\n')
            #f.write('dsharp1000um        Extension of name of dustkapscatmat_***.inp file\n')
            f.write('----------------------------------------------------------------------------\n')
        return
    if ngrain == 3:
        # Dust opacity control file  ----------------------------------
        with open('dustopac.inp','w+') as f:
            f.write('2               Format number of this file\n')
            f.write('1               Nr of dust species\n')
            f.write('============================================================================\n')
            if scat == 0: f.write('1               Way in which this dust species is read\n')    # 1 = kappa_*.inp  10 = kapscatmat_*.inp
            if scat == 1: f.write('10               Way in which this dust species is read\n')    # 1 = kappa_*.inp  10 = kapscatmat_*.inp
            f.write('0               0=Thermal grain\n')
            if scat == 0: f.write('amax10um_kaponly        Extension of name of dustkappa_***.inp file\n')
            if scat == 1: f.write('DSHARP_amax10.0um        Extension of name of dustkapscatmat_***.inp file\n')
            #f.write('dsharp_wav8.70e+02mic_amax1.50e+02mic        Extension of name of dustkappa_***.inp file\n')
            #f.write('dsharp1000um        Extension of name of dustkapscatmat_***.inp file\n')
            f.write('----------------------------------------------------------------------------\n')
        return


def write_velocity(vr,vtheta,vphi,vturb,nr,ntheta,nphi):
    # Write the gas velocity field   ----------------------------------
    with open('gas_velocity.inp','w+') as f:
        f.write('1\n')                       # Format number
        f.write('%d\n'%(nr*ntheta*nphi))     # Nr of cells
        for iphi in range(nphi):
            for itheta in range(ntheta):
                for ir in range(nr):
                    f.write('%13.6e %13.6e %13.6e\n'%(vr[ir,itheta,iphi],vtheta[ir,itheta,iphi],vphi[ir,itheta,iphi]))
    # Write the microturbulence file  ----------------------------------
    with open('microturbulence.inp','w+') as f:
        f.write('1\n')                       # Format number
        f.write('%d\n'%(nr*ntheta*nphi))     # Nr of cells
        data = vturb.ravel(order='F')        # Create a 1-D view, fortran-style indexing
        data.tofile(f, sep='\n', format="%13.6e")
        f.write('\n')
    return
    

def write_radmc_input(nphot,ngrain,scat,LTE=True):
    if ngrain == 1:
        # Write the radmc3d.inp control file   ----------------------------------
        with open('radmc3d.inp','w+') as f:
            f.write('nphot = %d\n'%(nphot))
            if scat == 0: f.write('scattering_mode_max = 0\n')
            if scat == 1: f.write('scattering_mode_max = 5\n')
            f.write('iranfreqmode = 1\n')
            f.write('tgas_eq_tdust = 1\n')       # Use the dust temperature of dust species 1 as gas temperature
            #f.write('rto_style = 3\n')   # Set binary T out
            #f.write('writeimage_unformatted = 1\n')  # Set binary image out
            if LTE==False:
                f.write('lines_mode=4')               # For perfoming the optically thin non-LTE transfer calculation    1 = LTE   / 4 = non-LTE
            else:
                f.write('lines_mode=1')
        return
    if ngrain == 2:
        # Write the radmc3d.inp control file   ----------------------------------
        with open('radmc3d.inp','w+') as f:
            f.write('nphot = %d\n'%(nphot))
            if scat == 0: f.write('scattering_mode_max = 0\n')
            if scat == 1: f.write('scattering_mode_max = 5\n')
            f.write('iranfreqmode = 1\n')
            #f.write('tgas_eq_tdust = 1\n')       # Use the dust temperature of dust species 1 as gas temperature
            #f.write('rto_style = 3\n')    # Set binary T out
            #f.write('writeimage_unformatted = 1\n')    # Set binary image out
            if LTE==False:
                f.write('lines_mode=4')               # For perfoming the optically thin non-LTE transfer calculation    1 = LTE   / 4 = non-LTE
            else:
                f.write('lines_mode=1')
        return


def write_Tdust(tdust,nr,ntheta,nphi):
    # Write the temperature file  ----------------------------------
    with open('dust_temperature.dat','w+') as f:
        f.write('1\n')                       # Format number
        f.write('%d\n'%(nr*ntheta*nphi))     # Nr of cells
        f.write('1\n')                       # Nr of dust species
        data = tdust.ravel(order='F')         # Create a 1-D view, fortran-style indexing
        data.tofile(f, sep='\n', format="%21.14e")
        f.write('\n')
    return
    
    
def write_Tgas(tgas,nr,ntheta,nphi):
    with open('gas_temperature.inp','w+') as f:
        f.write('1\n')                       # Format number
        f.write('%d\n'%(nr*ntheta*nphi))     # Nr of cells
        f.write('1\n')                       # Nr of dust species
        data = tgas.ravel(order='F')         # Create a 1-D view, fortran-style indexing
        data.tofile(f, sep='\n', format="%21.14e")
        f.write('\n')


def read_molecule_inp(mol):
    fnameread = 'molecule_'+mol+'.inp'
    with open(fnameread,'r') as f:
        s  = f.readline()
        molecule_name = f.readline().split()[0]
        s  = f.readline()
        molecule_weight = float(f.readline().split('\n')[0])
        s = f.readline()
        num_E = int(f.readline().split()[0])
        s = f.readline()
        level = np.zeros(num_E)
        g_factor = np.zeros(num_E)
        for ii in range(num_E):
            test =f.readline().split()
            level[ii] = float(test[0])
            g_factor[ii] = float(test[2])
        s = f.readline()
        num_tran = int(f.readline().split()[0])
        s = f.readline()
        up_level = np.zeros(num_tran)
        nu_ref = np.zeros(num_tran)
        E_u = np.zeros(num_tran)
        for ii in range(num_tran):
            test = f.readline().split()
            up_level[ii] = float(test[1])
            nu_ref[ii] = float(test[4])
            E_u[ii] = float(test[5])
        return level, g_factor, nu_ref, E_u, up_level
        

def therm_broadening(T,m):
    return np.sqrt(2*kb*T*np.log(2.0)/m)


def Partition_func(g, up_level, E_u, Tgas):
    Q = np.zeros_like(Tgas)
    for ii in range(len(E_u)):
        uid = int(up_level[ii])
        Q +=  g[uid-1]*np.exp(-E_u[ii]/Tgas)
    return Q
    

def line_tau(J,nu_ref,E_u,mu,T,Q,N,dV):
    # Mangum et al. 2017 ----------------------------------
    nu = nu_ref[J-1]*1e9
    a = h*nu/kb/T
    tau = (8*np.pi**3*J*mu**2/3/h/Q)*( np.exp(a)-1 )*np.exp(E_u[J-1]/T)*N/dV
    # (Goldsmith & Langer 1999, ApJ, 517, 209) ----------------------------------
    #a = h*B0/kb/Tgas_cyl
    #tau = (8*np.pi**3*mu**2/3/h)*NCO/dV/Q*J*np.exp(-a*J*(J+1))*(np.exp(2*a*J)-1)
    return tau


def emit_surface(r,z,tau,nr):
    layer = np.zeros(len(r))
    z_id = np.zeros(len(r))
    for i in range(len(r)):
        sum = 0.0
        for j in range(len(z)):
            sum += tau[i,j]
            if sum < 1.0:
                layer[i] = z[i,j]/r[i]
                z_id[i] = j
    new_r = np.linspace(r[0],r[-1],nr)
    intp_layer = inter.interp1d(r,layer,kind='nearest')
    new_layer = intp_layer(new_r)
    sph_d = np.sqrt(new_r**2*(1+np.tan(new_layer)**2))
    return sph_d, new_layer, z_id


def target_dust_opacity(size,obs_lamb):
    fnameread = 'dustkappa_amax'+size+'_kaponly.inp'
    with open(fnameread,'r') as f:
        s  = f.readline()
        nline  = int(f.readline().split()[0])
        s  = f.readline()
        lamb = np.zeros(nline)
        kap_abs = np.zeros(nline)
        for i in range(nline):
            temp = f.readline().split()
            lamb[i] = float(temp[0])
            kap_abs[i] = float(temp[1])
    intp_kap = inter.interp1d(lamb,kap_abs,kind='cubic')
    obs_kap = intp_kap(obs_lamb)
    return obs_kap
    

# Read temperature binary file
def read_T_binary(file):
    fnameread      = file
    hdr = np.fromfile(fnameread, count=4, dtype=int)
    if hdr[1] == 8:
        data = np.fromfile(fnameread, count=-1, dtype=np.float64)
    elif hdr[1] == 4:
        data = np.fromfile(fname, count=-1, dtype=float)
    data = np.reshape(data[4:], [ hdr[3], 1, 100, 100] )
    data = np.swapaxes(data, 0, 3)
    data = np.swapaxes(data, 1, 2)
    return data


# Read image binary file
def read_image_binary(file):
    fnameread      = 'image_'+file+'.bout'
    hdr = np.fromfile(fnameread, count=4, dtype=int)
    if hdr[1] == 8:
        data = np.fromfile(fnameread, count=-1, dtype=np.float64)
    elif hdr[1] == 4:
        data = np.fromfile(fname, count=-1, dtype=float)
    data = np.reshape(data[6+hdr[3]:], [ hdr[3], 1, 500, 500] )
    data = np.swapaxes(data, 0, 3)
    data = np.swapaxes(data, 1, 2)
    return data


def read_grid_inp(file):
    fnameread      = file
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
    return nr, ntheta, nphi, grids


def read_T_ascii(file):
    fnameread = file
    with open(fnameread,'r') as f:
        s         = f.readline()
        im_n      = int(f.readline())
        npop   = int(f.readline())
        temp = np.zeros(im_n*npop)
        for ilam in range(npop):
            for iy in range(im_n):
                temp[iy+(im_n*ilam)]  = f.readline()
    return im_n, npop, temp


def read_dens_inp(file):
    fnameread = file
    with open(fnameread,'r') as f:
        s         = f.readline()
        im_n      = int(f.readline())
        npop   = int(f.readline())
        rhod = np.zeros(im_n*npop)
        for ilam in range(npop):
            for iy in range(im_n):
                rhod[iy+(im_n*ilam)]  = f.readline()
    return im_n, npop, rhod


def read_image_ascii(file):
    fnameread = file
    with open(fnameread,'r') as f:
        s         = f.readline()
        im_n      = f.readline()
        im_nx     = int(im_n.split()[0])
        im_ny     = int(im_n.split()[1])
        nlam      = int(f.readline())
        pixsize   = f.readline()
        pixsize_x = float(pixsize.split()[0])
        pixsize_y = float(pixsize.split()[1])
        wavelength_mic = np.zeros(nlam)
        im    = np.zeros((nlam,im_ny,im_nx))
        rr_au = np.zeros((nlam,im_ny,im_nx))
        phi   = np.zeros((nlam,im_ny,im_nx))
        for ilam in range(nlam): wavelength_mic[ilam] = f.readline()
        for ilam in range(nlam):
            dummy = f.readline() #blank line
            for iy in range(im_ny):
                for ix in range(im_nx):
                    image_temp        = f.readline()
                    im[ilam,iy,ix]    = image_temp
                    rr_au[ilam,iy,ix] = np.sqrt(((ix-im_nx/2)*pixsize_x)**2+((iy-im_ny/2)*pixsize_y)**2)/au
                    phi[ilam,iy,ix]   = np.arctan2((iy-im_ny/2)*pixsize_y,(ix-im_nx/2)*pixsize_x)
        return im_nx,im_ny,nlam,pixsize_x,pixsize_y,im,rr_au,phi


def read_tau3D_ascii(file):
    fnameread = file
    with open(fnameread,'r') as f:
        s         = f.readline()
        im_n      = f.readline()
        im_nx     = int(im_n.split()[0])
        im_ny     = int(im_n.split()[1])
        nlam      = int(f.readline())
        wavelength_mic = np.zeros(nlam)
        im_x    = np.zeros((nlam,im_ny,im_nx))
        im_y    = np.zeros((nlam,im_ny,im_nx))
        im_z    = np.zeros((nlam,im_ny,im_nx))
        for ilam in range(nlam): wavelength_mic[ilam] = f.readline()
        for ilam in range(nlam):
            dummy = f.readline() #blank line
            for iy in range(im_ny):
                for ix in range(im_nx):
                    image_temp        = f.readline()
                    im_x[ilam,iy,ix]    = image_temp.split()[0]
                    im_y[ilam,iy,ix]    = image_temp.split()[1]
                    im_z[ilam,iy,ix]    = image_temp.split()[2]
    return im_nx,im_ny,nlam,im_x,im_y,im_z


def read_vel_input(file):
    fnameread = file
    with open(fnameread,'r') as f:
        s         = f.readline()
        im_n      = int(f.readline())
        vr        = np.zeros(im_n)
        vtheta    = np.zeros(im_n)
        vphi      = np.zeros(im_n)
        for iy in range(im_n):
            vel  = f.readline()
            vr[iy]   = vel.split()[0]
            vtheta[iy]   = vel.split()[1]
            vphi[iy]   = vel.split()[2]
    return vr, vtheta, vphi


def calculate_R0(rs,zs,t0,rt0,plt,mstar):
    eps=1e-3
    C = zs-rs
    r2 = rs;  r1 = 0.1*au;  r_mid = 0.5*(r1+r2)
    z2 = r2+C;  z1 = r1+C;  z_mid = r_mid+C
    t2 = t0*(r2/rt0)**plt;  t1 = t0*(r1/rt0)**plt;  t_mid = t0*(r_mid/rt0)**plt
    cs2 = np.sqrt(kb*t2/(2.3*mp));  cs1 = np.sqrt(kb*t1/(2.3*mp));  cs_mid = np.sqrt(kb*t_mid/(2.3*mp))
    omk2 = np.sqrt(GG*mstar/r2**3);  omk1 = np.sqrt(GG*mstar/r1**3);  omk_mid = np.sqrt(GG*mstar/r_mid**3)
    zb2 = 4.0*cs2/omk2;  zb1 = 4.0*cs1/omk1;  zb_mid = 4.0*cs_mid/omk_mid
    target2 = (zb2 - z2)/au;  target1 = (zb1 - z1)/au;  target_mid = (zb_mid - z_mid)/au
    if (target1*target2) > 0:
        R0 = 0.1* au
    else:
        while (0<1):
            if ( (target1*target_mid) <= 0 and (target2*target_mid) >= 0):
                r1=r1; r2=r_mid
            else:
                r1=r_mid; r2=r2
            r_mid = 0.5*(r1+r2)
            z2 = r2+C;  z1 = r1+C;  z_mid = r_mid+C
            if abs( (z2-z1)/(z2+z1) ) < eps :
                R0 = r_mid
                break
            t2 = t0*(r2/rt0)**plt;  t1 = t0*(r1/rt0)**plt;  t_mid = t0*(r_mid/rt0)**plt
            cs2 = np.sqrt(kb*t2/(2.3*mp));  cs1 = np.sqrt(kb*t1/(2.3*mp));  cs_mid = np.sqrt(kb*t_mid/(2.3*mp))
            omk2 = np.sqrt(GG*mstar/r2**3);  omk1 = np.sqrt(GG*mstar/r1**3);  omk_mid = np.sqrt(GG*mstar/r_mid**3)
            zb2 = 4.0*cs2/omk2;  zb1 = 4.0*cs1/omk1;  zb_mid = 4.0*cs_mid/omk_mid
            target2 = (zb2 - z2)/au;  target1 = (zb1 - z1)/au;  target_mid = (zb_mid - z_mid)/au
    return R0

