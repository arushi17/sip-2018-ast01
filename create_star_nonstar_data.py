#!/usr/bin/env python3
# program to sort all objects into star or nonstar directories based on zqual from fits files (for vdgc or halo7d)
# last modified: 7/13/18

from astropy.io import fits
import glob
import os.path
import re
import shutil
import sys

TARGET_DIR = os.path.expanduser("~") + '/ast01/training_data'

# Given a maskname variant and objname, returns the unique ID using the underlying maskname.
def unique_obj_id(mask_variant, objname):
    # HALO7D mask variations
    mask_map = {
            'E0c': 'E0',
            'E0d': 'E0',
            'E1a': 'E1',
            'E2a': 'E2',
            'E3a': 'E3',
            'E4a': 'E4',
            'E5a': 'E5',
            'E6a': 'E6',
            'E7a': 'E7',
            'GN0a': 'GN0',
            'GN2a': 'GN2',
            'GN3_cc': 'GN3',
            'GN3e_t': 'GN3',
            'GN3C': 'GN3',
            'GN3D': 'GN3',
            'c0c': 'c0',
            'c1a': 'c1',
            'c2a': 'c2',
            'c3a': 'c3',
            'gn1c': 'gn1',
            'gn1d': 'gn1',
            'gn1e_t': 'gn1',
            'gs0d': 'gs0',
            'gs0d_0': 'gs0',
            'gs1d_0': 'gs1',
            'gs1d': 'gs1',
    }

    true_mask = ''
    if mask_variant.startswith('vdgc'):
        true_mask = mask_variant
    elif mask_variant not in mask_map:
        print('Unknown HALO7D mask variant: {}'.format(mask_variant))
        sys.exit(1)
    else:
        true_mask = mask_map[mask_variant]

    return true_mask + '.' + objname

def copy_1_fits_file(source_data_dir, maskname, slitname, objname, zqual, target_dir):
    filename = "spec1d." + maskname + "." + '{:0>3}'.format(slitname) + "." + objname + ".fits"
    zqual = int(zqual)
    # Keep track of each object that we have known classification for, even if we are not
    # interested in each class. objname seems to be unique only within a mask.
    if zqual == 1 or zqual == 3 or zqual == 4:
        print('zqual ' + str(zqual) + ': ' + source_data_dir + '/' + filename + ' --> ' + target_dir + '/star')
        #shutil.copy(source_data_dir + '/' + filename, target_dir + '/star')
    if zqual == 0:
        print('zqual ' + str(zqual) + ': ' + source_data_dir + '/' + filename + ' --> ' + target_dir + '/nonstar')
        #shutil.copy(source_data_dir + '/' + filename, target_dir + '/nonstar')


def copy_from_1_directory_zspec_fits(survey_dir, fits_data, target_dir, known_zqual, do_not_copy_map):
    maskname = fits_data.field('MASKNAME')[0]
    print('\n++++++++ Processing mask: {} from zspec.fits'.format(maskname))
    
    zqual = fits_data.field('ZQUALITY')
    objname = fits_data.field('OBJNAME')
    slitname = fits_data.field('SLITNAME')
    
    print(slitname)
    print(objname)
    print(zqual)
    
    source_data_dir = survey_dir + '/' + maskname
    for i in range(len(zqual)):
        objid = unique_obj_id(maskname, objname[i])
        if objid in do_not_copy_map:
            continue
        known_zqual[objid] = int(zqual[i])
        copy_1_fits_file(source_data_dir, maskname, slitname[i], objname[i], zqual[i], target_dir)


# Also fills do_not_copy_map from the ppxf.dat file
def copy_from_1_directory_ppxf_path(survey_dir, ppxf_path, target_dir, known_zqual, do_not_copy_map):
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
        # print('line: ' + line)
        split_line_list = line.split()
        objname.append(split_line_list[0])
        slitname.append(split_line_list[1])
        zqual.append(split_line_list[6])

        if line.find('TiO') > -1 or line.find(' M ') > -1:
            do_not_copy_map[unique_obj_id(maskname, split_line_list[0])] = 0
            print('Found a TiO/M star, will exclude from dataset')
    
    print(slitname)
    print(objname)
    print(zqual)
    for i in range(len(zqual)):
        objid = unique_obj_id(maskname, objname[i])
        if objid in do_not_copy_map:
            continue
        known_zqual[objid] = int(zqual[i])
        copy_1_fits_file(source_data_dir, maskname, slitname[i], objname[i], zqual[i], target_dir)


def do_not_copy_map_from_txt(survey_dir, do_not_copy_map):
    # for halo7d only, checks all .txt files for mention of "TiO" or " M "
    # returns map of TiO/M objects
    txt_files = glob.glob(survey_dir + '/zspec/*notes.txt')
    for txt_file in txt_files:
        print(txt_file)

        with open(txt_file) as f:
            lines_list = f.readlines()
        for i in range(len(lines_list)):
            line = lines_list[i]
            split_line_list = line.split()
            if len(split_line_list) < 4:
                continue
            objname = split_line_list[1]
            maskname = os.path.basename(txt_file)[0:os.path.basename(txt_file).find('_')]
            
            if line.find('TiO') > -1 or line.find(' M ') > -1:
                do_not_copy_map[unique_obj_id(maskname, objname)] = 0
                print('Found a TiO/M star, will exclude from dataset')
            

# Process HALO7D survey
def process_halo7d_survey(survey_dir, target_dir, known_zqual, do_not_copy_map):
    do_not_copy_map_from_txt(survey_dir, do_not_copy_map)
    zspec_fits = glob.glob(survey_dir + '/zspec/zspec*.fits')
    for fits_file in zspec_fits:
        zspec = fits.open(fits_file)
        #zspec.info()
        data = zspec[1].data
    
        print(fits_file)
        copy_from_1_directory_zspec_fits(survey_dir, data, target_dir, known_zqual, do_not_copy_map)

    ppxf_list = glob.glob(survey_dir + '/zspec/*_ppxf.dat')
    for ppxf_file in ppxf_list:
        print(ppxf_file)
        copy_from_1_directory_ppxf_path(survey_dir, ppxf_file, target_dir, known_zqual, do_not_copy_map)

    # All halo7d files NOT mentioned in zspec.fits or ppxf.dat are nonstars 
    extra_dirs = [survey_dir + '/GN3_cc']  # No zqual for these, but they are confirmed galaxies
    extra_dirs.extend(glob.glob(survey_dir + '/deimos_spr14/*'))

    for source_data_dir in extra_dirs:
        print('\n++++++++ Processing extra dir: {}'.format(source_data_dir))
        all_files = glob.glob(source_data_dir + '/spec1d*.fits')
        for filepath in all_files:
            filename = os.path.basename(filepath)
            m = re.search('spec1d\.(.+)\.[0-9]+\.(.+)\.fits', filename)
            maskname = m.group(1)
            objname = m.group(2)
            if objname.startswith('serendip'):
                continue  # We don't trust that these serendipitous objects are galaxies

            objid = unique_obj_id(maskname, objname)
            if objid not in known_zqual and objid not in do_not_copy_map:
                known_zqual[objid] = 0  # Galaxy
                print('Not in zspec.fits: ' + filepath + ' --> ' + target_dir + '/nonstar')
                #shutil.copy(filepath, target_dir + '/nonstar')


# Process VDGC survey
def process_vdgc_survey(survey_dir, target_dir, known_zqual, do_not_copy_map):
    zspec_fits = glob.glob(survey_dir + '/zspec/zspec*.fits')
    for fits_file in zspec_fits:
        zspec = fits.open(fits_file)
        #zspec.info()
        data = zspec[1].data
    
        print(fits_file)
        # TODO: Should we exclude serendip from VDGC? We do have ZQual for them (most nonstars).
        copy_from_1_directory_zspec_fits(survey_dir, data, target_dir, known_zqual, do_not_copy_map)


# All halo7d files NOT mentioned in zspec.fits or ppxf.dat are nonstars.
# Use this map to keep track of them.
known_zqual = {}  # Empty dict
do_not_copy_map = {}  # Empty dict

process_vdgc_survey(os.path.expanduser("~") + '/ast01/vdgc', TARGET_DIR, known_zqual, do_not_copy_map)
process_halo7d_survey(os.path.expanduser("~") + '/ast01/halo7d', TARGET_DIR, known_zqual, do_not_copy_map)

print('\n++++++++ SUMMARY')
# Invert the map so we can print summary stats
inv_map = {}
for k, v in known_zqual.items():
    inv_map.setdefault(v, []).append(k)
for k, v in inv_map.items():
    print('\nZQual {}: {} spectra\n{}'.format(k, len(v), sorted(v)))
