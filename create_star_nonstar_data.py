#!/usr/bin/env python3
# program to sort all objects into star or nonstar directories based on zqual from fits files (for vdgc or halo7d)
# last modified: 7/13/18

from astropy.io import fits
import glob
import os.path
import re
import shutil

TARGET_DIR = os.path.expanduser("~") + '/ast01/training_data'

def copy_1_fits_file(source_data_dir, maskname, slitname, objname, zqual, target_output, copied):
    filename = "spec1d." + maskname + "." + '{:0>3}'.format(slitname) + "." + objname + ".fits"
    zqual = int(zqual)
    copied[filename] = 1
    if zqual == 1 or zqual == 4:
        print('zqual ' + str(zqual) + ': ' + source_data_dir + '/' + filename + ' --> ' + target_output + '/star')
        shutil.copy(source_data_dir + '/' + filename, target_output + '/star')
    if zqual == 0:
        print('zqual ' + str(zqual) + ': ' + source_data_dir + '/' + filename + ' --> ' + target_output + '/nonstar')
        shutil.copy(source_data_dir + '/' + filename, target_output + '/nonstar')


def copy_from_1_directory_zspec_fits(survey_dir, fits_data, target_output):
    maskname = fits_data.field('MASKNAME')[0]
    print('\n++++++++ Processing mask: {} from zspec.fits'.format(maskname))
    
    zqual = fits_data.field('ZQUALITY')
    objname = fits_data.field('OBJNAME')
    slitname = fits_data.field('SLITNAME')
    
    print(slitname)
    print(objname)
    print(zqual)
    
    source_data_dir = survey_dir + '/' + maskname
    copied = {}  # Empty dict
    for i in range(len(zqual)):
        copy_1_fits_file(source_data_dir, maskname, slitname[i], objname[i], zqual[i], target_output, copied)

    # All halo7d files NOT mentioned in zspec.fits are nonstars 
    if os.path.basename(survey_dir) == 'halo7d':
        print(source_data_dir + '/spec1d*.fits')
        all_files = glob.glob(source_data_dir + '/spec1d*.fits')
        for filepath in all_files:
            filename = os.path.basename(filepath)
            if filename not in copied:
                print('Not in zspec.fits: ' + filepath + ' --> ' + target_output + '/nonstar')
                shutil.copy(filepath, target_output + '/nonstar')


def copy_from_1_directory_ppxf_path(survey_dir, ppxf_path, target_output):
    # Extract maskname from filename
    maskname = re.search(r"(.+?)_ppxf.dat", os.path.basename(ppxf_path)).group(1)
    print('\n++++++++ Processing mask: {} from ppxf.dat'.format(maskname))

    source_data_dir = survey_dir + '/' + maskname

    slitname = []
    objname = []
    zqual = []
    
    with open(ppxf_path) as f:
        lines_list = f.readlines()
    
    for i in range(len(lines_list)):
        line = lines_list[i]
        print('line: ' + line)
        split_line_list = line.split()
        objname.append(split_line_list[0])
        slitname.append(split_line_list[1])
        zqual.append(split_line_list[6])
    
    print(slitname)
    print(objname)
    print(zqual)
    copied = {}
    for i in range(len(zqual)):
        copy_1_fits_file(source_data_dir, maskname, slitname[i], objname[i], zqual[i], target_output, copied)


# Process a survey, such as vdgc or halo7d.
def process_survey(survey_dir, target_dir):
    zspec_fits = glob.glob(survey_dir + '/zspec/zspec*.fits')
    for fits_file in zspec_fits:
        zspec = fits.open(fits_file)
        #zspec.info()
        data = zspec[1].data
    
        print(fits_file)
        copy_from_1_directory_zspec_fits(survey_dir, data, target_dir)

    ppxf_list = glob.glob(survey_dir + '/zspec/*_ppxf.dat')
    for ppxf_file in ppxf_list:
        print(ppxf_file)
        copy_from_1_directory_ppxf_path(survey_dir, ppxf_file, target_dir)


process_survey(os.path.expanduser("~") + '/ast01/halo7d', TARGET_DIR)
process_survey(os.path.expanduser("~") + '/ast01/vdgc', TARGET_DIR)
