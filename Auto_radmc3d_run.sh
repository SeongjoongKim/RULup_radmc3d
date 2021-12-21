#/bin/sh
# the meaning of name:
# fiducial = DSHARP_H0.1_D0.1_S0.001_3e6
# wind_fiducial = wind_DSHARP_H0.1_D0.1_S0.001_3e6
name=fiducial

#--------------------------------------------------------------------------------------------
# Setup the input files for temperature calculation
python problem_setup.py -calmode "T" -wind "F"  # Set 2 grain model for calculating Tdust & Tgas
# Calculate the dust temperature
TIMESTAMP=`date +%Y-%m-%d_%H-%M-%S`
echo $TIMESTAMP
radmc3d mctherm setthreads 4
TIMESTAMP=`date +%Y-%m-%d_%H-%M-%S`
echo $TIMESTAMP

cp dust_temperature.dat "Tdust_"$name".dat"      # Copy T file

#--------------------------------------------------------------------------------------------
# Setup the input files for imaging
python make_gas_temperature.py -file $name      # Separate Tdust & Tgas
python problem_setup.py -calmode "I" -wind "F"  # Set large grain only for imaging
#'''
#--------------------------------------------------------------------------------------------
# Continuum
TIMESTAMP=`date +%Y-%m-%d_%H-%M-%S`
echo $TIMESTAMP
#radmc3d image setthreads 4 noline lambda 1300.4036557964412 npix 500 sizeau 800 incl 25.0 posang 239.0 #nphot_scat 100000
#mv image.out image_cont12_21.out
#radmc3d image setthreads 4 noline lambda 881.1004345450963 npix 500 sizeau 800 incl 25.0 posang 239.0 #nphot_scat 100000
#mv image.out image_contCN.out
#radmc3d image setthreads 4 noline lambda 1360.2279857533088 npix 500 sizeau 800 incl 25.0 posang 239.0 #nphot_scat 100000
#mv image.out image_cont13_21.out
#radmc3d image setthreads 4 noline lambda 906.8462549748294 npix 500 sizeau 800 incl 25.0 posang 239.0 #nphot_scat 100000
#mv image.out image_cont13_32.out
radmc3d image setthreads 4 noline lambda 1362.8404254642091 npix 500 sizeau 800 incl 25.0 posang 239.0 #nphot_scat 100000
mv image.out image_cont220GHz.out
#radmc3d image setthreads 4 noline lambda 910.3086709358545 npix 500 sizeau 800 incl 25.0 posang 239.0 #nphot_scat 100000
#mv image.out image_cont18_32.out
TIMESTAMP=`date +%Y-%m-%d_%H-%M-%S`
echo $TIMESTAMP
radmc3d image setthreads 4 noline lambda 891.505 npix 500 sizeau 800 incl 25.0 posang 239.0 #nphot_scat 100000
mv image.out image_cont336GHz.out
#--------------------------------------------------------------------------------------------
# Line emission
TIMESTAMP=`date +%Y-%m-%d_%H-%M-%S`
echo $TIMESTAMP
#radmc3d calcpop setthreads 4
radmc3d image imolspec 1 iline 2 incl 25.0 posang 239.0 setthreads 4 widthkms 6.0 linenlam 200 npix 500 sizeau 800 #doppcatch   #nphot_scat 400000 #
mv image.out image_12CO_21.out
TIMESTAMP=`date +%Y-%m-%d_%H-%M-%S`
echo $TIMESTAMP
#radmc3d calcpop setthreads 4
radmc3d image imolspec 4 iline 42,44,47 incl 25.0 posang 239.0 setthreads 4 widthkms 6.0 linenlam 200 npix 500 sizeau 800 #doppcatch   #nphot_scat 400000 #
mv image.out image_CN_32.out
TIMESTAMP=`date +%Y-%m-%d_%H-%M-%S`
echo $TIMESTAMP
#radmc3d calcpop setthreads 4
radmc3d image imolspec 2 iline 2 incl 25.0 posang 239.0 setthreads 4 widthkms 6.0 linenlam 200 npix 500 sizeau 800 #doppcatch   #nphot_scat 400000 #
mv image.out image_13CO_21.out
TIMESTAMP=`date +%Y-%m-%d_%H-%M-%S`
echo $TIMESTAMP
radmc3d image imolspec 2 iline 3 incl 25.0 posang 239.0 setthreads 4 widthkms 6.0 linenlam 200 npix 500 sizeau 800 #doppcatch   #nphot_scat 400000 #
mv image.out image_13CO_32.out
TIMESTAMP=`date +%Y-%m-%d_%H-%M-%S`
echo $TIMESTAMP
#radmc3d calcpop setthreads 4
radmc3d image imolspec 3 iline 2 incl 25.0 posang 239.0 setthreads 4 widthkms 6.0 linenlam 200 npix 500 sizeau 800 #doppcatch   #nphot_scat 400000 #
mv image.out image_C18O_21.out
TIMESTAMP=`date +%Y-%m-%d_%H-%M-%S`
echo $TIMESTAMP
radmc3d image imolspec 3 iline 3 incl 25.0 posang 239.0 setthreads 4 widthkms 6.0 linenlam 200 npix 500 sizeau 800 #doppcatch   #nphot_scat 400000 #
mv image.out image_C18O_32.out
TIMESTAMP=`date +%Y-%m-%d_%H-%M-%S`
echo $TIMESTAMP
#--------------------------------------------------------------------------------------------
# converting ascii result file to fits
python FITS_convert.py -file $name -dist 160.0
python FITS_convert.py -file $name -dist 160.0 -bmaj 0.51 -bmin 0.44 -bpa 80.0
#'''

#'''
name=fiducial
python problem_setup.py -calmode "T" -wind "F"  # Set 2 grain model for calculating Tdust & Tgas
python plot_disk_rz.py -file $name               # Plot the disk temperature and density distribution in r-z plane

python problem_setup.py -calmode "I" -wind "F"  # Set 2 grain model for calculating Tdust & Tgas
# Plotting image maps
python plot_maps.py -file $name"_bmaj51" -dist 160.0
python plot_maps.py -file $name"_bmaj5" -dist 160.0

#--------------------------------------------------------------------------------------------
# Do manually after running the script
# Making azimuthally averaged profile by IDL
# idl
# .r RadialProfile, radplot
# RadialProfile
python azimuthal_average.py -file $name -bmaj "bmaj5"
python azimuthal_average.py -file $name -bmaj "bmaj51"

#--------------------------------------------------------------------------------------------
# Ploting radial profiles
python plot_radial.py -file $name"_bmaj51"
python plot_radial.py -file $name"_bmaj5"

#--------------------------------------------------------------------------------------------
# Ploting spectrum at each pixel
python plot_spectra.py -mole "CN_3-2" -bmaj "bmaj51" -double "T"
python plot_spectra.py -mole "CN_3-2" -bmaj "bmaj5" -double "T"
python plot_spectra.py -mole "CN_3-2" -bmaj "bmaj51" -double "F"
python plot_spectra.py -mole "CN_3-2" -bmaj "bmaj5" -double "F"
python plot_spectra.py -mole "C18O_2-1" -bmaj "bmaj51" -double "F"
python plot_spectra.py -mole "C18O_2-1" -bmaj "bmaj5" -double "F"

python plot_spectra_average.py -mole "C18O_2-1" -bmaj "bmaj51" -double "F"
python plot_spectra_average.py -mole "C18O_2-1" -bmaj "bmaj5" -double "F"
python plot_spectra_average.py -mole "CN_3-2" -bmaj "bmaj51" -double "F"
python plot_spectra_average.py -mole "CN_3-2" -bmaj "bmaj5" -double "F"
python plot_spectra_average.py -mole "CN_3-2" -bmaj "bmaj51" -double "T"
python plot_spectra_average.py -mole "CN_3-2" -bmaj "bmaj5" -double "T"

#--------------------------------------------------------------------------------------------
# Ploting Moment 1 and 2 maps of the models
mole=CN_3-2
tname=fiducial
python plot_moments.py -mole $mole -bmaj "bmaj51" -tname $tname
python plot_moments.py -mole $mole -bmaj "bmaj5" -tname $tname

#--------------------------------------------------------------------------------------------
# Ploting subtractions of Moment 1 and 2 maps of two specific molecules (mole1-mole2)
mole1=CN_3-2
mole2=C18O_2-1
tname=fiducial_wind
python plot_moments_sub.py -mole1 $mole1 -mole2 $mole2 -bmaj "bmaj51" -tname $tname
python plot_moments_sub.py -mole1 $mole1 -mole2 $mole2 -bmaj "bmaj5" -tname $tname

#'''
mole="CN_3-2"
python plot_spectra_LWtest.py -mole $mole -bmaj "bmaj51" -double "T" -rmin 0.05 -rmax 0.15 -PAmin 45 -PAmax 135
python plot_spectra_LWtest.py -mole $mole -bmaj "bmaj5" -double "T" -rmin 0.05 -rmax 0.15 -PAmin 45 -PAmax 135
python plot_spectra_LWtest.py -mole $mole -bmaj "bmaj51" -double "T" -rmin 0.15 -rmax 0.25 -PAmin 45 -PAmax 135
python plot_spectra_LWtest.py -mole $mole -bmaj "bmaj5" -double "T" -rmin 0.15 -rmax 0.25 -PAmin 45 -PAmax 135
python plot_spectra_LWtest.py -mole $mole -bmaj "bmaj51" -double "T" -rmin 0.25 -rmax 0.35 -PAmin 45 -PAmax 135
python plot_spectra_LWtest.py -mole $mole -bmaj "bmaj5" -double "T" -rmin 0.25 -rmax 0.35 -PAmin 45 -PAmax 135
python plot_spectra_LWtest.py -mole $mole -bmaj "bmaj51" -double "T" -rmin 0.35 -rmax 0.45 -PAmin 45 -PAmax 135
python plot_spectra_LWtest.py -mole $mole -bmaj "bmaj5" -double "T" -rmin 0.35 -rmax 0.45 -PAmin 45 -PAmax 135
python plot_spectra_LWtest.py -mole $mole -bmaj "bmaj51" -double "T" -rmin 0.45 -rmax 0.55 -PAmin 45 -PAmax 135
python plot_spectra_LWtest.py -mole $mole -bmaj "bmaj5" -double "T" -rmin 0.45 -rmax 0.55 -PAmin 45 -PAmax 135
python plot_spectra_LWtest.py -mole $mole -bmaj "bmaj51" -double "T" -rmin 0.55 -rmax 0.65 -PAmin 45 -PAmax 135
python plot_spectra_LWtest.py -mole $mole -bmaj "bmaj5" -double "T" -rmin 0.55 -rmax 0.65 -PAmin 45 -PAmax 135
python plot_spectra_LWtest.py -mole $mole -bmaj "bmaj51" -double "T" -rmin 0.65 -rmax 0.75 -PAmin 45 -PAmax 135
python plot_spectra_LWtest.py -mole $mole -bmaj "bmaj5" -double "T" -rmin 0.65 -rmax 0.75 -PAmin 45 -PAmax 135
python plot_spectra_LWtest.py -mole $mole -bmaj "bmaj51" -double "T" -rmin 0.75 -rmax 0.85 -PAmin 45 -PAmax 135
python plot_spectra_LWtest.py -mole $mole -bmaj "bmaj5" -double "T" -rmin 0.75 -rmax 0.85 -PAmin 45 -PAmax 135


mole="C18O_2-1"
python plot_spectra_average_LWtest.py -mole $mole -bmaj "bmaj51" -double "F" -rmin 0.05 -rmax 0.15 -PAmin 45 -PAmax 135
python plot_spectra_average_LWtest.py -mole $mole -bmaj "bmaj5" -double "F" -rmin 0.05 -rmax 0.15 -PAmin 45 -PAmax 135
python plot_spectra_average_LWtest.py -mole $mole -bmaj "bmaj51" -double "F" -rmin 0.15 -rmax 0.25 -PAmin 45 -PAmax 135
python plot_spectra_average_LWtest.py -mole $mole -bmaj "bmaj5" -double "F" -rmin 0.15 -rmax 0.25 -PAmin 45 -PAmax 135
python plot_spectra_average_LWtest.py -mole $mole -bmaj "bmaj51" -double "F" -rmin 0.25 -rmax 0.35 -PAmin 45 -PAmax 135
python plot_spectra_average_LWtest.py -mole $mole -bmaj "bmaj5" -double "F" -rmin 0.25 -rmax 0.35 -PAmin 45 -PAmax 135
python plot_spectra_average_LWtest.py -mole $mole -bmaj "bmaj51" -double "F" -rmin 0.35 -rmax 0.45 -PAmin 45 -PAmax 135
python plot_spectra_average_LWtest.py -mole $mole -bmaj "bmaj5" -double "F" -rmin 0.35 -rmax 0.45 -PAmin 45 -PAmax 135
python plot_spectra_average_LWtest.py -mole $mole -bmaj "bmaj51" -double "F" -rmin 0.45 -rmax 0.55 -PAmin 45 -PAmax 135
python plot_spectra_average_LWtest.py -mole $mole -bmaj "bmaj5" -double "F" -rmin 0.45 -rmax 0.55 -PAmin 45 -PAmax 135
python plot_spectra_average_LWtest.py -mole $mole -bmaj "bmaj51" -double "F" -rmin 0.55 -rmax 0.65 -PAmin 45 -PAmax 135
python plot_spectra_average_LWtest.py -mole $mole -bmaj "bmaj5" -double "F" -rmin 0.55 -rmax 0.65 -PAmin 45 -PAmax 135
python plot_spectra_average_LWtest.py -mole $mole -bmaj "bmaj51" -double "F" -rmin 0.65 -rmax 0.75 -PAmin 45 -PAmax 135
python plot_spectra_average_LWtest.py -mole $mole -bmaj "bmaj5" -double "F" -rmin 0.65 -rmax 0.75 -PAmin 45 -PAmax 135
python plot_spectra_average_LWtest.py -mole $mole -bmaj "bmaj51" -double "F" -rmin 0.75 -rmax 0.85 -PAmin 45 -PAmax 135
python plot_spectra_average_LWtest.py -mole $mole -bmaj "bmaj5" -double "F" -rmin 0.75 -rmax 0.85 -PAmin 45 -PAmax 135
