import numpy as np
import pdb
from astropy.coordinates import SkyCoord, EarthLocation
from astropy.time import Time
import astropy.units as u
from astropy.io import fits

class Spectrum(object):
#Create a Spectrum object from a spec1d file. 
#Read file and store wavelength, flux, ivar, and MJD
    def __init__(self, filename):
        hdu=fits.open(filename)
        blue=hdu[1].data
        red=hdu[2].data
        lam_blue=blue.field('lambda') #Returns array of dims (1, 4096); hence the clugey reshaping
        spec_blue=blue.field('spec')
        ivar_blue=blue.field('ivar')
        lam_red=red.field('lambda')
        spec_red=red.field('spec')
        ivar_red=red.field('ivar')
        skyblue=blue.field('skyspec')
        skyred=red.field('skyspec')
        lam=np.concatenate((lam_blue, lam_red), axis=1)
        lam=lam.reshape((8192))
        spec=np.concatenate((spec_blue, spec_red), axis=1)
        spec=spec.reshape((8192))
        ivar=np.concatenate((ivar_blue, ivar_red), axis=1)
        ivar=ivar.reshape((8192))
        sky=np.concatenate((skyblue, skyred), axis=1)
        sky=sky.reshape((8192))
        #Toss lambda > 9000
#        keep=(lam < 9000)
        self.lam=lam
        self.flux=spec
        self.ivar=ivar
        self.sky=sky
        self.mjd=float(hdu[1].header['mjd-obs'])
        self.coords=SkyCoord(hdu[1].header['ra'], hdu[1].header['dec'], unit=(u.hourangle, u.deg), frame="icrs")
        self.vhelio=get_vhelio(self.mjd, self.coords)
        hdu.close()

    def mask_spec(self, mask):
        self.lam=self.lam[mask]
        self.flux=self.flux[mask]
        self.ivar=self.ivar[mask]

#    def get_vhelio(self):
def get_vhelio(mjd, coords):
        #Returns the heliocentric correction for an observation. Uses the Time.light_travel_time
        #to compute the time barycentric correction for each observation. This time is converted to a distance,
        #and the velocity is measured by computing this distance over 30 mins ahead/behind the observation.
    c=3.e5*u.km/u.s
    keck=EarthLocation(-5464487.817598869*u.m, -2492806.5910856915*u.m, 2151240.1945184576*u.m)
#    keck=EarthLocation.of_site('keck')
    dt=(1.*u.hour).to(u.day)
#        times_array=[self.mjd-(dt.value)/2.,self.mjd+(dt.value)/2.]*u.day
    times_array=[mjd-(dt.value)/2., mjd+(dt.value)/2.]*u.day
    times=Time(times_array.value, format='mjd', scale='utc', location=keck)
#        ltt=times.light_travel_time(self.coords, 'heliocentric')
    ltt=times.light_travel_time(coords, 'heliocentric')
    dist=(c*ltt)
#        pdb.set_trace()
    vhelio=-1.*(np.diff(dist)/np.diff(times_array)).to(u.km/u.s)
    return vhelio.value

class Spectrum_Fake(object):
    def __init__(self, lam, flux, ivar, mjd, coords):
        self.lam=lam
        self.flux=flux
        self.ivar=ivar
        self.mjd=mjd
        self.coords=coords
        self.vhelio=get_vhelio(self.mjd, self.coords)

    def mask_spec(self, mask):
        self.lam=self.lam[mask]
        self.flux=self.flux[mask]
        self.ivar=self.ivar[mask]

class Template(object):
    #Class storing wavelength, flux and velocity for Template. Velocities measured independently. 
    def __init__(self, filename, velocity):
        temp=Spectrum(filename)
#        data=np.genfromtxt(filename, names=['lam', 'spec'])
        self.lam = temp.lam
        self.flux = temp.flux
        self.v=velocity

def test_spec():
    spec_file=['/u/ecunning/bayesian_velocities/E4_676/23mar15/spec1d.E4a.014.676.fits.gz', \
                   '/u/ecunning/bayesian_velocities/E4_676/24mar15/spec1d.E4a.014.676.fits.gz', \
                   '/u/ecunning/bayesian_velocities/E4_676/25mar15/spec1d.E4a.014.676.fits.gz', \
                   '/u/ecunning/bayesian_velocities/E4_676/12apr15/spec1d.E4b.014.676.fits.gz', \
                   '/u/ecunning/bayesian_velocities/E4_676/13apr15/spec1d.E4b.014.676.fits.gz', \
                   '/u/ecunning/bayesian_velocities/E4_676/18may15/spec1d.E4a.014.676.fits.gz', \
                   '/u/ecunning/bayesian_velocities/E4_676/19may15/spec1d.E4a.014.676.fits.gz']
    for i in range(len(spec_file)):
        spec=Spectrum(spec_file[i])
        print(spec.vhelio)
