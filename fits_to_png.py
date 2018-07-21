#!/usr/bin/env python3
# program to convert .fits files to .png after gaussian smoothing
# last modified: 7/6/18

from spectrum_class import Spectrum
from pandas import Series
from matplotlib import pyplot
from scipy import ndimage
import argparse
import math
import sys
from spectrum_fits_utils import *

# output path based on input path
def outputPath(input_path, smooth_type, smooth_val, plot_type):
    file = input_path[:-5]
    if smooth_val == 'none':
        # default, normal spectrum
        if plot_type == 'flux':
            output = file + '_rawFlux.png'
        elif plot_type == 'ivar':
            output = file + '_rawIvar.png'
        else:
            output = file + '_rawFluxIvar.png'
    elif smooth_type == 'adaptiveSmoothing':
        output = file + '_' + smooth_type + '.png'
    else:
        output = file + '_' + smooth_type + str(smooth_val) + '.png'
    return output

# uses spectrum_fits_utils to apply gaussian smoothing to flux array, save it as a series object with .png
def gaussianAndSave(input_path, outputPath, smooth_type, smooth_val, plot_type):
    print('input path: ' + input_path, 'output path: ' + outputPath, 'smoothing type: ' + smooth_type, 'smoothing value: ' + str(smooth_val)) # TESTING
    spec = Spectrum(input_path)
    fig = pyplot.figure()
    
    # uses other functions to perform user's specified smoothing
    if smooth_type == 'sigma' and smooth_val != 'none':
        # generates spectrum with regular gaussian smoothing, sigma specified
        series = regularSigmaGaussianAndSave(spec, plot_type, smooth_val)
        series.plot(figsize=(10,6))
    elif smooth_type == 'adaptive':
        # generates spectrum with adaptive smoothing on flux
        series = adaptiveGaussianAndSave(spec, smooth_val)
        series.plot(figsize=(10,6))
    elif smooth_type == 'window':
        # generates spectrum with adaptive smoothing on flux with fixed window size
        series = fixedWindowAdaptiveGaussianAndSave(spec, smooth_val)
        series.plot(figsize=(10,6))
    else:    
        # default, generates normal raw spectrum
        series, series_ivar = defaultGaussianAndSave(spec, plot_type, fig)
        series_ivar.plot()
        series.plot(figsize=(10,6))
        #TODO: fig.legend((series_ivar, series), ('ivar', 'flux'), 'upper left')

    # plots main series and saves png
    pyplot.savefig(outputPath)
    pyplot.close(fig)

# gaussianAndSave function for default parameters, generates raw spectrum
def defaultGaussianAndSave(spec, plot_type, fig):
    if plot_type == 'flux':
        series = Series(spec.flux, spec.lam)
    elif plot_type == 'ivar':
        series = Series(spec.ivar, spec.lam)
    else:
        series = Series(scale(cleanAndLimit(spec.flux, 0), 0, 1), spec.lam)
        series_ivar = Series(scale(cleanAndLimit(spec.ivar, 2.5), 0, 1), spec.lam)
    return series, series_ivar

# gaussianAndSave function for regular gaussian smoothing, sigma specified
def regularSigmaGaussianAndSave(spec, plot_type, smooth_val):
    cleaned_flux, cleaned_ivar = cleanFluxIvar(spec.flux, spec.ivar)
    if plot_type == 'flux':
        smoothed_flux = gaussianSmoothing(int(smooth_val), cleaned_flux)
        series = Series(smoothed_flux, spec.lam)
    elif plot_type == 'ivar':
        smoothed_ivar = gaussianSmoothing(int(smooth_val), cleaned_ivar)
        series = Series(smoothed_ivar, spec.lam)
    else:
        series = Series(smoothed_flux, spec.lam)
        series_ivar = Series(smoothed_ivar, spec.lam)
        series_ivar.plot()
    return series

# gaussianAndSave function for adaptive smoothing on flux
def adaptiveGaussianAndSave(spec, smooth_val):
    print('adaptiveGaussianAndSave reached') # testing
    smoothed_flux = adaptiveSmoothing(spec.flux, spec.ivar, smooth_val)
    series = Series(smoothed_flux, spec.lam)
    return series

# gaussianAndSave function for adaptive smoothing on flux with fixed window sizE
def fixedWindowAdaptiveGaussianAndSave(spec, smooth_val):
    smoothed_flux = adaptiveSmoothing(spec.flux, spec.ivar, smooth_val) 
    series = Series(smoothed_flux, spec.lam)
    return series

# defines parser arguments and gets user input, returns args
def parserInput():
    parser = argparse.ArgumentParser(description='Convert .fits files to .png files with regular gaussian smoothing or adaptive smoothing.')
    parser.add_argument('filepath', nargs='+', help='Path to .fits files to be converted to .png')
    parser.add_argument('-s', '--sigma', type=str, default='none', help='Sigma to be used in regular gaussian smoothing, "none" (default) if no smoothing.')
    parser.add_argument('-a', '--adaptive', type=str, default='none', help='"adaptive" if adaptive smoothing, "none" (default) if no adaptive smoothing, only performed on flux.')
    parser.add_argument('-w', '--window', type=str, default='0', help='"15", "31", or "61" set window for adaptive smoothing, "0" (default) if no specified window.')
    parser.add_argument('-p', '--plotType', type=str, default='flux', help='"flux" (default)  if only flux plotted, "ivar" if only ivar plotted, "both" if both plotted (will be scaled).')
    args = parser.parse_args()
    print(args) # TESTING
    return args

# checks that the arguments are valid (sigma, window, and adaptive are not all specified simultaneously)
def validArgs(args):
    if args.sigma != 'none' and args.adaptive != 'none':
        print('Cannot specify both sigma and adaptive smoothing.')
        sys.exit(1)
    if args.sigma != 'none' and args.window != '0':
        print('Cannot specify both sigma and window size.')
        sys.exit(1)
    if args.adaptive != 'none' and args.window != '0':
        print('Cannot specify both adaptive smoothing and window size.')
        sys.exit(1)
    if args.window != '0' and args.window != '15' and args.window != '31' and args.window != '61':
        print('Cannot specify window sizes other than 0, 15, 31, and 61.')
        sys.exit(1)
    if args.sigma == 0:
        print('Must specify a sigma greater than 0.')
        sys.exit(1)

# Now this can be used as a library, and also run as a top-level script.
if __name__=='__main__':
    args = parserInput()
    validArgs(args)

    # loop over all input files, perform smoothing and save to png
    count = 0
    for filename in args.filepath:
        count = count + 1 # testing
        if args.sigma != 'none':
            print('sigma') # testing
            output = outputPath(filename, 'sigma', args.sigma, args.plotType)
            gaussianAndSave(filename, output, 'sigma', args.sigma, args.plotType)
        elif args.adaptive == 'adaptive':
            print('adaptive') # testing
            output = outputPath(filename, 'adaptiveSmoothing', args.adaptive, args.plotType)
            gaussianAndSave(filename, output, 'adaptive', args.adaptive, args.plotType)
        elif args.window != '0':
            print('window') # testing
            output = outputPath(filename, 'windowSize', args.window, args.plotType)
            gaussianAndSave(filename, output, 'window', args.window, args.plotType)
        else:
            # if none of the parameters are specified other than default, generate raw spectrum
            print('default') # testing
            output = outputPath(filename, 'sigma', 'none', args.plotType)
            gaussianAndSave(filename, output, 'sigma', 'none', args.plotType)

        print('figures still open: ' + str(pyplot.get_fignums())) # TESTING

    print('number of fits files converted: ' + str(count)) # TESTING