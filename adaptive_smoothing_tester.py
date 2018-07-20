# tests adaptive smoothing 
# last modified: 7/6/18

from spectrum_fits_utils import *
from spectrum_class import Spectrum
import numpy as np

input = '/Users/vikram/ast01/halo7d/E1a/spec1d.E1a.003.28680.fits'
output = '/Users/vikram/ast01/halo7d/E1a/spec1d.E1a.003.28680_smoothedFlux.png'
output_scaled = '/Users/vikram/ast01/halo7d/E1a/spec1d.E1a.003.28680_scaledIvar.png'
output_raw = '/Users/vikram/ast01/halo7d/E1a/spec1d.E1a.003.28680_rawIvar.png'

spec = Spectrum(input)
flux_in = spec.flux
ivar_in = spec.ivar

smoothed_flux = adaptiveSmoothing(flux_in, ivar_in)
savePngToFile(spec.lam, smoothed_flux, output)
