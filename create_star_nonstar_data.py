#!/usr/bin/env python3
# program to sort all objects into star or nonstar directories based on zqual from fits files (for vdgc or halo7d)
# last modified: 7/13/18

from astropy.io import fits
import errno
import glob
import os.path
import os
import re
import shutil
import sys

##################### IMPORTANT! SET THIS TO THE CORRECT DATA VERSION #######################
DATA_VERSION=v7


# Given a maskname variant and objname, returns the unique ID using the underlying maskname.
def unique_obj_id(mask_variant, objname):
    # HALO7D mask variations
    mask_map = {
            'E0': 'E0',
            'E0c': 'E0',
            'E0d': 'E0',
            'E1': 'E1',
            'E1a': 'E1',
            'E2': 'E2',
            'E2a': 'E2',
            'E3': 'E3',
            'E3a': 'E3',
            'E4': 'E4',
            'E4a': 'E4',
            'E5a': 'E5',
            'E5b': 'E5',
            'E5s': 'E5',
            'E5s_1': 'E5',
            'E5s_2': 'E5',
            'E5shy': 'E5',
            'E6': 'E6',
            'e6': 'E6',
            'E6_s': 'E6',
            'E6a': 'E6',
            'E6s': 'E6',
            'E6b': 'E6',
            'E7a': 'E7',
            'E7b': 'E7',
            'E7_s': 'E7',
            'GN0a': 'GN0',
            'GN0b': 'GN0',
            'GN0c': 'GN0',
            'GN2a': 'GN2',
            'GN2b': 'GN2',
            'GN3_cc': 'GN3',
            'GN3e_t': 'GN3',
            'GN3C': 'GN3',
            'GN3D': 'GN3',
            'c0': 'c0',
            'c0c': 'c0',
            'c1a': 'c1',
            'c1b': 'c1',
            'c1c': 'c1',
            'c1d': 'c1',
            'c1s': 'c1',
            'c2a': 'c2',
            'c2b': 'c2',
            'c2c': 'c2',
            'c2s': 'c2',
            'c3a': 'c3',
            'c3b': 'c3',
            'c3s': 'c3',
            'gn1': 'gn1',
            'gn1c': 'gn1',
            'gn1d': 'gn1',
            'gn1e_t': 'gn1',
            'gs0d': 'gs0',
            'gs0a': 'gs0',
            'gs0b': 'gs0',
            'gs0d_0': 'gs0',
            'gs0a_0': 'gs0',
            'gs0a_1': 'gs0',
            'gs0b_2': 'gs0',
            'gs0e_1': 'gs0',
            'gs1d_0': 'gs1',
            'gs1b_2': 'gs1',
            'gs1c_0': 'gs1',
            'gs1c_1': 'gs1',
            'gs1e_1': 'gs1',
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
    if zqual == 1 or zqual == 4:
        print('zqual ' + str(zqual) + ': ' + source_data_dir + '/' + filename + ' --> ' + target_dir + '/star')
        shutil.copy(source_data_dir + '/' + filename, target_dir + '/star')
    if zqual == 0:
        print('zqual ' + str(zqual) + ': ' + source_data_dir + '/' + filename + ' --> ' + target_dir + '/nonstar')
        shutil.copy(source_data_dir + '/' + filename, target_dir + '/nonstar')


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


def do_not_copy_map_candels_halo7d_catalog(survey_dir, map_candels_halo7d_catalog):
    # for the auto-add halo7d galaxies: do not add those with zspec -99
    catalog = survey_dir + '/zspec/ALL4_CANDELS_HALO7D_catalog.txt'
    print(catalog)

    with open(catalog) as f:
        lines_list = f.readlines()
    for i in range(len(lines_list)):
        line = lines_list[i]
        split_line_list = line.split()
        if len(split_line_list) < 9:
            continue
        objname = split_line_list[1]
        zspec = split_line_list[8]

        if zspec == '-99.0':
            map_candels_halo7d_catalog[objname] = 0

def do_not_copy_map_from_txt_out(survey_dir, do_not_copy_map):
    # for halo7d only, checks all .txt files for mention of "TiO" or " M "
    # also checks all .out files for "-2 0" pattern
    
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
    
    out_files = glob.glob(survey_dir + '/zspec/*.out')
    for out_file in out_files:
        print(out_file)

        with open(out_file) as f:
            lines_list = list(f)
        for i in range(len(lines_list)):
           line = lines_list[i]
           split_line_list = line.split()
           if len(split_line_list) < 8:
               continue
           objname = split_line_list[0]
           maskname = os.path.basename(out_file)[0:os.path.basename(out_file).find('.')]
           objid = unique_obj_id(maskname, objname)

           al_star_6 = split_line_list[6]
           al_star_7 = split_line_list[7]
           if al_star_6 == '-2' and al_star_7 == '0':
               if objid not in do_not_copy_map:
                    do_not_copy_map[objid] = 0
                    print('{}: Found an alignment star, will exclude from dataset'.format(objid))
            

# Process HALO7D survey
def process_halo7d_survey(survey_dir, target_dir, known_zqual, do_not_copy_map, map_candels_halo7d_catalog):
    do_not_copy_map_from_txt_out(survey_dir, do_not_copy_map)
    do_not_copy_map_candels_halo7d_catalog(survey_dir, map_candels_halo7d_catalog)
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
            if objname in map_candels_halo7d_catalog:
                print('{}: Found an unknown object in CANDELS/HALO7D catalog, will exclude from dataset'.format(objname))
                continue # Unknown objects with -99.0 zspec
            if objname.startswith('serendip'):
                print('{}: Found a serendip object, will exclude from dataset'.format(objname))
                continue  # We don't trust that these serendipitous objects are galaxies

            objid = unique_obj_id(maskname, objname)
            if objid not in known_zqual and objid not in do_not_copy_map:
                known_zqual[objid] = 0  # Galaxy
                print('Not in zspec.fits: ' + filepath + ' --> ' + target_dir + '/nonstar')
                shutil.copy(filepath, target_dir + '/nonstar')


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


def make_dirs(root_dir):
    print('Making directories in {}'.format(root_dir))
    os.makedirs(root_dir, exist_ok=True)
    os.makedirs(root_dir + '/star', exist_ok=True)
    os.makedirs(root_dir + '/nonstar', exist_ok=True)


if __name__=='__main__':
    # Make all the directories
    training_dir = os.path.expanduser("~") + '/ast01/data.' + DATA_VERSION + '/training_data'
    dev_dir = os.path.expanduser("~") + '/ast01/data.' + DATA_VERSION + '/dev_data'
    test_dir = os.path.expanduser("~") + '/ast01/data.' + DATA_VERSION + '/test_data'
    make_dirs(training_dir)
    make_dirs(dev_dir)
    make_dirs(test_dir)

    # All halo7d files NOT mentioned in zspec.fits or ppxf.dat are nonstars.
    # Use this map to keep track of them.
    known_zqual = {}  # Empty dict
    do_not_copy_map = {}  # Empty dict
    map_candels_halo7d_catalog = {} # Empty dict

    process_vdgc_survey(os.path.expanduser("~") + '/ast01/vdgc', training_dir, known_zqual, do_not_copy_map)
    process_halo7d_survey(os.path.expanduser("~") + '/ast01/halo7d', training_dir, known_zqual, do_not_copy_map, map_candels_halo7d_catalog)

    print('\n++++++++ SUMMARY')
    # Invert the map so we can print summary stats
    inv_map = {}
    for k, v in known_zqual.items():
        inv_map.setdefault(v, []).append(k)
    for k, v in inv_map.items():
        print('\nZqual {}: {} spectra\n{}'.format(k, len(v), sorted(v)))
