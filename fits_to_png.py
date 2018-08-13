#!/usr/bin/env python3
# program to convert .fits files to .png after gaussian smoothing
# last modified: 7/6/18

from spectrum_class import Spectrum
from pandas import Series
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot
from scipy import ndimage
import argparse
import math
import os.path
import sys
from spectrum_fits_utils import *

# Truncation of x-axis while using log(lam) binning:
MIN_BIN_LAMBDA=5200
MAX_BIN_LAMBDA=9500

FLUX_COLOR='blue'
IVAR_COLOR='orange'
LINE_WIDTH=0.5

OUTPUT_DIR = os.path.expanduser("~") + '/ast01/graphs/'

# output path based on input path
# smooth_type should be adaptiveSmoothing, logbin, windowSize, or sigma.
# smooth_val should be either a (str) odd whole number or 'none' or 'adaptive'.
# plot_type should be flux, ivar, or both.
def outputPath(input_path, smooth_type, smooth_val, plot_type):
    obj_type = os.path.basename(os.path.dirname(input_path))  # star or nonstar
    filename = os.path.basename(input_path)[:-5]  # Remove the .fits extension

    if smooth_val == 'none':
        # default, normal spectrum
        if plot_type == 'flux':
            output = filename + '_rawFlux.png'
        elif plot_type == 'ivar':
            output = filename + '_rawIvar.png'
        else:
            output = filename + '_scaledFluxIvar.png'
    elif smooth_type == 'adaptiveSmoothing':
        output = filename + '_' + smooth_type + '.png'
    elif smooth_type == 'logbin':
        output = filename + '_LogBin' + smooth_val + '.png'
    else:
        output = filename + '_' + smooth_type + str(smooth_val) + '.png'
    return OUTPUT_DIR + obj_type + '/' + output

# uses spectrum_fits_utils to apply gaussian smoothing to flux array, save it as a series object with .png
# smooth_type should be adaptive, logbin, window, or sigma.
def gaussianAndSave(input_path, outputPath, smooth_type, smooth_val, plot_type):
    print('input path: ' + input_path, 'output path: ' + outputPath, 'smoothing type: ' + smooth_type, 'smoothing value: ' + str(smooth_val)) # TESTING
    spec = Spectrum(input_path)
    fig = pyplot.figure(figsize=(10,6))
    
    # uses other functions to perform user's specified smoothing
    if smooth_type == 'sigma' and smooth_val != 'none':
        # generates spectrum with regular gaussian smoothing, sigma specified
        series = regularSigmaGaussian(spec, plot_type, smooth_val)
        series.plot(label='flux', color=FLUX_COLOR, linewidth=LINE_WIDTH)
        pyplot.title('{} - {}-sigma Gaussian smoothing'.format(os.path.basename(input_path), smooth_val))
    elif smooth_type == 'adaptive':
        # generates spectrum with adaptive smoothing on flux
        series = adaptiveGaussian(spec, smooth_val)
        series.plot(label='flux', color=FLUX_COLOR, linewidth=LINE_WIDTH)
        pyplot.title('{} - adaptive Gaussian smoothing'.format(os.path.basename(input_path)))
    elif smooth_type == 'window':
        # generates spectrum with adaptive smoothing on flux with fixed window size
        series = adaptiveGaussian(spec, smooth_val)
        series.plot(label='flux', color=FLUX_COLOR, linewidth=LINE_WIDTH)
        pyplot.title('{} - {}-pixel Gaussian smoothing'.format(os.path.basename(input_path), smooth_val))
    else:    
        # default, generates raw series chart for either flux, ivar, or both
        series_flux, series_ivar = cleanedSeries(spec, plot_type)
        if not series_ivar.empty and not series_flux.empty:
            series_ivar.plot(label='ivar', color=IVAR_COLOR, linewidth=LINE_WIDTH)
            series_flux.plot(label='flux', color=FLUX_COLOR, linewidth=LINE_WIDTH)
            pyplot.title('{} - unsmoothed flux and ivar (scaled)'.format(os.path.basename(input_path)))
            #fig.legend((series_ivar, series_flux), ('ivar', 'flux'), 'upper left')
        elif not series_flux.empty:
            series_flux.plot(label='flux', color=FLUX_COLOR, linewidth=LINE_WIDTH)
            pyplot.title('{} - flux'.format(os.path.basename(input_path)))
            # pyplot.ylabel('flux') # Useful only if we have units
        else:
            series_ivar.plot(label='ivar', color=IVAR_COLOR, linewidth=LINE_WIDTH)
            pyplot.title('{} - ivar'.format(os.path.basename(input_path)))

    # Common plot settings
    pyplot.legend()
    pyplot.grid(True)
    pyplot.gca().xaxis.grid(True) # Vertical lines
    pyplot.gca().xaxis.grid(True, which='minor') # Vertical lines
    # Visually present the data on log scale (by default pyplot will make it linear)
    pyplot.xlabel('lambda (Angstrom)')

    pyplot.savefig(outputPath)
    pyplot.close(fig)


# gaussian function for default parameters, generates raw spectrum
def cleanedSeries(spec, plot_type):
    series_flux = Series([])
    series_ivar = Series([])
    flux, ivar = cleanFluxIvar(spec.flux, spec.ivar)
    if plot_type == 'flux':
        series_flux = Series(flux, spec.lam)
    elif plot_type == 'ivar':
        series_ivar = Series(ivar, spec.lam)
    else:
        # Since we are plotting both on one, need to scale them to fit.
        series_flux = Series(scale(flux, 0, 1), spec.lam)
        series_ivar = Series(scale(ivar, 0, 1), spec.lam)
    return series_flux, series_ivar


# gaussian function for regular gaussian smoothing, sigma specified
def regularSigmaGaussian(spec, plot_type, smooth_val):
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

# gaussian function for adaptive smoothing on flux
# smooth_val should be either a (str) odd whole number or 'adaptive'.
def adaptiveGaussian(spec, smooth_val):
    print('adaptiveGaussian reached') # testing
    smoothed_flux = adaptiveSmoothing(spec.flux, spec.ivar, smooth_val)
    series = Series(smoothed_flux, spec.lam)
    return series


# crude function for plotting flux of a spectrum with adjusted pixels on log scale
def logBinPixPlot(num_pix, input_path, output_path):
    num_pix = int(num_pix)
    spec = Spectrum(input_path)
    fig = pyplot.figure(figsize=(10,6))
    smoothed_flux = adaptiveSmoothing(spec.flux, spec.ivar)
    flux, lam = logBinPixels(num_pix, spec.lam, smoothed_flux, min_lam=MIN_BIN_LAMBDA, max_lam=MAX_BIN_LAMBDA, restore_lam_scale=True)
    series = Series(flux, lam)
    series.plot(label='flux', color=FLUX_COLOR, linewidth=LINE_WIDTH)
    pyplot.legend()
    pyplot.grid(True)
    pyplot.gca().xaxis.grid(True) # Vertical lines
    pyplot.gca().xaxis.grid(True, which='minor') # Vertical lines
    # Visually present the data on log scale (by default pyplot will make it linear)
    pyplot.title('{} - {} bins of log(lambda)'.format(os.path.basename(input_path), num_pix))
    pyplot.xlabel('lambda (Angstrom)')
    pyplot.xscale('log')
    pyplot.savefig(output_path)
    pyplot.close(fig)

# defines parser arguments and gets user input, returns args
def parserInput():
    parser = argparse.ArgumentParser(description='Convert .fits files to .png files with regular gaussian smoothing or adaptive smoothing.')
    parser.add_argument('filepath', nargs='+', help='Path to .fits files to be converted to .png')
    parser.add_argument('-s', '--sigma', type=str, default='none', help='Sigma to be used in regular gaussian smoothing, "none" (default) if no smoothing.')
    parser.add_argument('-a', '--adaptive', type=str, default='none', help='"adaptive" if adaptive smoothing, "none" (default) if no adaptive smoothing, only performed on flux.')
    parser.add_argument('-w', '--window', type=str, default='0', help='"15", "31", or "61" set window for adaptive smoothing, "0" (default) if no specified window.')
    parser.add_argument('-p', '--plotType', type=str, default='flux', help='"flux" (default)  if only flux plotted, "ivar" if only ivar plotted, "both" if both plotted (will be scaled 0-1).')
    parser.add_argument('-l', '--logbin', type=str, default='0', help='Pixel size to adjust image to, using binning on a log lambda scale.')
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
    if args.sigma == '0':
        print('Must specify a sigma greater than 0.')
        sys.exit(1)
    if int(args.logbin) < 0:
        print('Must specify a pixel value greater than or equal to 0')
        sys.exit(1)

# Now this can be used as a library, and also run as a top-level script.
if __name__=='__main__':
    args = parserInput()
    validArgs(args)

    # loop over all input files, perform smoothing and save to png
    count = 0
    for filename in args.filepath:
        count = count + 1 # testing
        print('Processing {}'.format(filename))
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
        elif args.logbin != '0':
            print('logbin image size: {}'.format(args.logbin)) # testing
            output = outputPath(filename, 'logbin', args.logbin, args.plotType)
            logBinPixPlot(args.logbin, filename, output)
        else:
            # if none of the parameters are specified other than default, generate raw spectrum
            print('default') # testing
            output = outputPath(filename, 'sigma', 'none', args.plotType)
            gaussianAndSave(filename, output, 'sigma', 'none', args.plotType)

    print('number of fits files converted: ' + str(count)) # TESTING
