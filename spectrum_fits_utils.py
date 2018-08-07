# library for working with numpy arrays representing spectra from fits files
# last modified: 7/6/18

from spectrum_class import Spectrum
from pandas import Series
from matplotlib import pyplot
from scipy import ndimage
import math
import numpy as np

# given an x axis, y axis, and filepath, saves a png
def savePngToFile(x_series, y_series, filepath):
    fig = pyplot.figure()
    series = Series(y_series, x_series)
    series.plot(figsize=(10,6))
    pyplot.savefig(filepath)
    pyplot.close(fig)

# apply gaussian smoothing to flux array, returns smoothed spectrum
# spec: spectrum, from spec = Spectrum(input_path)
# sigma: controls amount of smoothing
# array: either flux or ivar
def gaussianSmoothing(sigma, array): 
    gaussian_array = ndimage.filters.gaussian_filter1d(array, sigma, truncate=4.0)
    return gaussian_array

# returns weights of gaussian windows of 15, 30, 60, and 120 for adaptive smoothing
# window_size: size of desired window
def getGaussianWindow(window_size):
    if window_size == 15:
        return np.array([0.009033, 0.018476, 0.033851, 0.055555, 0.08167, 0.107545, 0.126854, 0.134032, 0.126854, 0.107545, 0.08167, 0.055555, 0.033851, 0.018476, 0.009033])
    elif window_size == 31:
        return np.array([0.009091, 0.0114, 0.014073, 0.017104, 0.020466, 0.02411, 0.027962, 0.031928, 0.035893, 0.039724, 0.043284, 0.046433, 0.04904, 0.05099, 0.052198, 0.052607, 0.052198, 0.05099, 0.04904, 0.046433, 0.043284, 0.039724, 0.035893, 0.031928, 0.027962, 0.02411, 0.020466, 0.017104, 0.014073, 0.0114, 0.009091])
    elif window_size == 61:
        return np.array([0.00999, 0.010473, 0.010961, 0.011454, 0.01195, 0.012448, 0.012946, 0.013442, 0.013934, 0.014422, 0.014903, 0.015375, 0.015837, 0.016286, 0.016722, 0.017142, 0.017544, 0.017927, 0.018289, 0.018629, 0.018944, 0.019235, 0.019498, 0.019733, 0.01994, 0.020116, 0.020261, 0.020375, 0.020457, 0.020506, 0.020522, 0.020506, 0.020457, 0.020375, 0.020261, 0.020116, 0.01994, 0.019733, 0.019498, 0.019235, 0.018944, 0.018629, 0.018289, 0.017927, 0.017544, 0.017142, 0.016722, 0.016286, 0.015837, 0.015375, 0.014903, 0.014422, 0.013934, 0.013442, 0.012946, 0.012448, 0.01195, 0.011454, 0.010961, 0.010473, 0.00999])
    else:
        return np.array([0])

# Make sure each value is a valid positive float. If not, replace with a 0.
# Returns a cleaned numpy array.
def cleanValues(series_in):
    series = np.array(series_in) 
    for i in range(len(series)):
        try:
            n = float(series[i])
        except ValueError:
            series[i] = 0
        if series[i] < 0 or math.isnan(series[i]) or math.isinf(series[i]):
            series[i] = 0
    return series
    

# limit_outliers, if > 1, specifies that outliers will be truncated to this many stddev
# Returns a new numpy array.
def limitOutliers(series_in, limit_outliers):
    series = np.array(series_in) 
    if limit_outliers <= 1:
        return series

    stddev = np.std(series)
    mean = np.mean(series)
    max_val = mean + limit_outliers * stddev
    min_val = mean - limit_outliers * stddev
    for i in range(len(series)):
        if series[i] > max_val:
            series[i] = max_val
        elif series[i] < min_val:
            series[i] = min_val
    return series

# returns a new numpy array that standardizes an already cleaned series
def standardize(series):
    mean = np.mean(series)
    stddev = np.std(series)
    std_series = (series - mean) / stddev
    return std_series

# returns a new numpy array that scales an already cleaned series to be inbetween max and min (expects series to be a numpy array)
def scale(series, min_scale, max_scale):
    series_min = series.min()
    series_max = series.max()
    return ((series - series_min) / (series_max - series_min)) * (max_scale - min_scale) + min_scale

# returns window size
def determineWindowSize(ivar, i, smooth_val):
    if smooth_val == 'adaptive':
        # 15 pixel window
        avg_ivar = np.mean(ivar[i-7:i+8])
        if avg_ivar >= 0.4:
            return 15

        # 31 pixel window
        avg_ivar = np.mean(ivar[i-15:i+16])
        if avg_ivar >= 0.3:
            return 31

        # 61 pixel window
        return 61
    else:
        # pre-specified window size 
        return int(smooth_val)

# returns cleaned up flux and ivar
def cleanFluxIvar(flux_in, ivar_in):
    ivar = cleanValues(ivar_in)
    ivar = np.sqrt(ivar)
    ivar = limitOutliers(ivar, 2.5)
    ivar = scale(ivar, 0, 1)

    flux = cleanValues(flux_in)
    flux = scale(flux, 0, 1)

    return flux, ivar

#TODO: limit scaling to reasonable range
# returns a numpy array storing the combination of gaussian weights and ivar, scaled
def gaussianWeightedIvar(window_size, ivar, i):
    gweights = getGaussianWindow(window_size)
    slice_min = int(i - ((window_size-1) / 2))
    slice_max = int(i + ((window_size+1) / 2))
    if (slice_max - slice_min != window_size):
        print('ERROR: window_size: {}, i: {}, slice_min: {}, slice_max: {}'.format(window_size, i, slice_min, slice_max))
    ivar_slice = ivar[slice_min:slice_max]
    weighted_ivar = np.multiply(gweights, ivar_slice)
    sum_of_weights = np.sum(weighted_ivar)
    if math.isnan(sum_of_weights) or sum_of_weights == 0:
        # Forget about modulating by ivar values.
        # TODO: Should output flux be zero instead, if we can't trust it?
        # return np.zeros(window_size)
        return gweights
    else:
        return weighted_ivar / sum_of_weights

# returns a single adapted flux value
def convolutionFlux(flux, window_size, weighted_ivar, i):
    step_size = (window_size - 1) / 2
    slice_min = int(i - step_size)
    slice_max = int(i + step_size + 1)
    if (slice_max - slice_min != window_size):
        print('ERROR: window_size: {}, i: {}, slice_min: {}, slice_max: {}'.format(window_size, i, slice_min, slice_max))
    flux_slice = flux[slice_min:slice_max]
    return np.asscalar(np.convolve(np.flip(flux_slice, axis=0), weighted_ivar, mode='valid'))

# returns a numpy array containing smoothed flux, size 8k
def adaptiveSmoothing(flux_in, ivar_in, smooth_val='adaptive'):
    flux, ivar = cleanFluxIvar(flux_in, ivar_in)
    ivar = np.pad(ivar, (30, 30), 'edge')
    flux = np.pad(flux, (30, 30), 'edge')
    
    histogram = {15: 0, 31: 0, 61: 0}
    smoothed_flux = np.zeros(len(flux_in))

    for i in range(30, len(flux) - 30):
        window_size = determineWindowSize(ivar, i, smooth_val)
        histogram[window_size] = histogram[window_size] + 1
        weighted_ivar = gaussianWeightedIvar(window_size, ivar, i)
        weighted_ivar_sum = np.sum(weighted_ivar)
        if not math.isclose(1.0, weighted_ivar_sum, rel_tol=0.02):
            print('Warning: weighted ivar of {} != 1.0'.format(weighted_ivar_sum))
        smoothed_flux[i - 30] = convolutionFlux(flux, window_size, weighted_ivar, i)

    print('histogram of window sizes: ')
    print(histogram)

    return smoothed_flux

# returns a numpy array containing spectrum wavelength adjusted to log scale
def logLamConvert(lam):
    loglam = np.zeros(len(lam))
    for i in range(len(lam)):
        if lam[i] <= 0:
            loglam[i] = 0
        else:
            loglam[i] = np.log(lam[i])
    return loglam

# returns flux and wavelengths of a spectrum converted to a desired number of pixels,
# with x-axis binned by lambda adjusted to log scale. So, at the blue end of the spectrum
# we may have 30 pixels binned into a single output pixel, while at the red end we may
# have 60. Effectively stretches the spectrum at the blue end, shrinks at the red end.
# The returned log(lambda) values represent the upper limit of each bin.
# Best to use this after adaptive smoothing.
# Since there is likely some benefit of fixing log(lam) to x-axis position/pixel mapping
# across all spectra, caller can supply non-zero min_lam and max_lam values (in Angtroms).
# Wavelengths below min_lam (and their flux) will be dropped; similarly for max_lam.
def logBinPixels(num_pix, lam, flux, min_lam=0, max_lam=0):
    # determine how many log(lam) each pixel in adjusted image should cover
    # Find the first non-zero lambda value at each end.
    if min_lam > 0:
        min_loglam = math.log(min_lam)
    else:
        for i in range(len(lam)):
            if lam[i] > 0:
                min_loglam = math.log(lam[i])
                break
    if max_lam > 0:
        max_loglam = math.log(max_lam)
    else:
        for i in range(len(lam)-1, -1, -1):
            if lam[i] > 0:
                max_loglam = math.log(lam[i])
                break

    # create array of boundaries for pixel bins in terms of log(lam), stores upper boundary
    pix_width = (max_loglam - min_loglam) / num_pix
    bin_bound = np.zeros(num_pix)
    for i in range(num_pix):
        bin_bound[i] = min_loglam + pix_width * (i + 1)

    # go through lam array and fill pixel bins by taking mean of all below the boundary
    adj_spec = np.zeros(num_pix)
    lam_end_pix = 0
    for bound in range(num_pix):
        lam_start_pix = lam_end_pix
        while lam_end_pix < len(lam) and lam[lam_end_pix] <= math.exp(bin_bound[bound]): 
            lam_end_pix += 1
        # print('bound: {}, lam_start_pix: {}, lam_end_pix: {}'.format(bound, lam_start_pix, lam_end_pix)) # TESTING
        if lam_end_pix > lam_start_pix:
            # lam_end_pix now points to 1 beyond the range we want
            adj_spec[bound] = np.mean(flux[lam_start_pix:lam_end_pix])
            # print('adj_spec: {}'.format(adj_spec[bound])) # TESTING
    return np.float32(adj_spec), np.float32(bin_bound)

