#!/usr/bin/env python

import sys
import inspect

import numpy as np
import nibabel as nib

import scipy.ndimage as sn
import scipy.ndimage.interpolation as sni
import pickle
import pdb

def upsample(niiFileName, upsampledFile, zoom_values_file='upsampling_log.pickle', isotrop_res=True, upsample_factor=None, polynomial='3'):
    """
    Upsample a nifti and save the upsampled image.
    The upsampling procedure has been implemented in a way that it is easily revertable.

    Example arguments:
    niiFileName = '10529_t1.nii.gz'
    upsampled_file = '10529_t1_upsampled.nii.gz'
    """

    # load the nifti
    nii = nib.load(niiFileName)
    header = nii.header
    affine = nii.affine

    # make data type of image to float 
    out_dtype = np.float32
    header['datatype'] = 16 # corresponds to float32
    header['bitpix'] = 32 # corresponds to float32

    # in case nothing should be done
    isotrop_res = bool(int(isotrop_res))
    if ((not isotrop_res) and (upsample_factor is None)):
        print('Uspampling not requested. Skipping...')
        nii.to_filename(upsampledFile)
        return nii

    # convert input to number
    if isotrop_res:
        isotrop_res = float(np.min(header.get_zooms()[0:3]))
        all_upsampling = [float(zoom)/isotrop_res for zoom in header.get_zooms()[0:3]]
        for idx, zoom in enumerate(all_upsampling):
            if zoom<1:
                all_upsampling[idx]= 1.
            else:
                all_upsampling[idx]= np.round(zoom)
    else:
        upsample_factor = float(upsample_factor)
        all_upsampling = [upsample_factor for zoom in header.get_zooms()[0:3]]
        

    polynomial = int(polynomial)

    old_volume = np.squeeze(nii.get_fdata().astype(float))
    # get new volume shape and update upsampling based on rounded numbers
    old_shape = old_volume.shape
    print(old_shape)
    new_shape = tuple([np.round(old_shape[ii]*usampling).astype(int) for ii, usampling in enumerate(all_upsampling)])
    print(new_shape)

    # upsample image
    print('Upsampling volume...')
    vol = sn.zoom(old_volume, all_upsampling)

    print('Done.')

    # update voxel sizes in header
    if len(header.get_zooms())==3:
        new_zooms = tuple( [header.get_zooms()[ii]/float(all_upsampling[ii]) for ii in np.arange(3)] ) # 3 spatial dimensions
    elif len(header.get_zooms())>3:
        tmp = [header.get_zooms()[ii]/float(all_upsampling[ii]) for ii in np.arange(3)]
        tmp.extend(list(header.get_zooms()[3:]))
        new_zooms = tuple(tmp) # 3 spatial dimensions + 1 time
    else:
        print('Cannot handle this stuff... ')
        print(header.get_zooms())
        raise Exception('Header has less than 2 entries. 2D?')

    header.set_zooms(new_zooms)

    # adapt affine according to scaling
    all_upsampling.extend([1.]) # time
    scaling = np.diag(1./np.asarray(all_upsampling))
    affine = np.dot(affine, scaling)

    # create new NII
    newNii = nib.Nifti1Image(vol.astype(out_dtype), header=header, affine=affine)

    # save niftis
    newNii.to_filename(upsampledFile)

    # save upsampling factors
    with open(zoom_values_file, 'wb') as outfile:
        pickle.dump([all_upsampling[:-1],polynomial, old_shape], outfile)


    return (newNii)

def downsample(niiFileName, downsampled_file, zoom_values_file='upsampling_log.pickle', order=3):
    """
    downsample a nifti which has been upsampled with the function above.

    Example arguments:
    niiFileName = '10529_t1_upsampled.nii.gz'
    downsample_file = '10529_t1_downsample.nii.gz'
    zoom_values_file = 'upsampling_log.pickle'
    """

    # load the nifti
    nii = nib.load(niiFileName)
    header = nii.header

    # make data type of image to float 
    out_dtype = np.float32
    header['datatype'] = 16 # corresponds to float32
    header['bitpix'] = 32 # corresponds to float32
    
    downsample_factor=[]
    with open(zoom_values_file,'rb') as zfile:
        [all_upsampling, polynomial, old_shape] = pickle.load(zfile)

    print('Downsampling with scales: ' + str(1./np.asarray(all_upsampling)))
    if old_shape:
        current_shape = nii.get_fdata().shape
        print(old_shape)
        downsample_values = np.asarray([float(old_shape[ii])/float(current_shape[ii]) for ii in np.arange(len(old_shape))])
    else:
        if len(all_upsampling) == 1:
            downsample_values = 1./np.asarray(3*all_upsampling)
        else:
            downsample_values = 1./np.asarray(all_upsampling)

    # downsampling image
    print('Downsampling volume...')
    vol = sn.zoom(nii.get_fdata(), downsample_values, order=int(order))
    print('Done.')

    # update voxel sizes in header
    if len(header.get_zooms())==3:
        new_zooms = tuple( [header.get_zooms()[ii]/float(downsample_values[ii]) for ii in np.arange(3)] ) # 3 spatial dimensions
    elif len(header.get_zooms())>3:
        tmp = [header.get_zooms()[ii]/float(downsample_values[ii]) for ii in np.arange(3)]
        tmp.extend(list(header.get_zooms()[3:]))
        new_zooms = tuple(tmp) # 3 spatial dimensions + 1 time
    else:
        print('Cannot handle this stuff... ')
        print(header.get_zooms())
        raise Exception('Header has less than 2 entries. 2D?')

    # new_zooms = tuple( [header.get_zooms()[ii]/float(downsample_values[ii]) for ii in np.arange(len(header.get_zooms()))] )
    header.set_zooms(new_zooms)

    # adapt affine according to scaling
    affine = nii.affine
    downsample_values = downsample_values.tolist()
    downsample_values.extend([1.]) # time
    scaling = np.diag(1./np.asarray(downsample_values))
    affine = np.dot(affine, scaling)

    # create new NII
    newNii = nib.Nifti1Image(vol.astype(out_dtype), header=header, affine=affine)

    # save niftis
    newNii.to_filename(downsampled_file)

    return (newNii)


def binarize(infile, outfile, threshold, infofile=None, name=None):

    # load file
    nii = nib.load(infile)
    vol = nii.get_fdata()
    vol = (vol>=float(threshold)).astype(int)

    # create new NII
    newNii = nib.Nifti1Image(vol, header=nii.header, affine=nii.affine)

    # save niftis
    if outfile != "None":
        newNii.to_filename(outfile)

    if infofile is not None:
        voxel_size = np.prod(np.asarray(nii.header.get_zooms()).astype(float))
        volume=np.sum(vol)*voxel_size
        with open(infofile, 'a') as fid:
            fid.write('%s: %f \n' %(name, volume))

    return (newNii)

functions = {
        'binarize': binarize,
        'upsample': upsample,
        'downsample': downsample
        }

def print_help(func, name):
    signature = '{script} {name} {args}'.format(
            script=sys.argv[0],
            name=name,
            args=' '.join(inspect.getfullargspec(func).args))
    print(signature)
    print(func.__doc__)

if __name__ == '__main__':
    usage = 'USAGE: \n{script} <command> <arguments>\n    Commands: [ {commands} ]\n{script} -h <command>'.format(
            script=sys.argv[0],
            commands = ' | '.join(functions.keys()),
            )
    if len(sys.argv) == 1:
        print(usage)
        sys.exit(1)

    command = sys.argv[1]
    # Handle help
    if command == '-h':
        if len(sys.argv) == 2:
            print(usage)
        else:
            func_name = sys.argv[2]
            if func_name not in functions:
                print("Unknown command " + func_name)
                sys.exit(1)
            func = functions[func_name]
            print_help(func, func_name)
        sys.exit(0)

    # Handle normal usage
    arguments = sys.argv[2:]
    func = functions[command]
    argspec = inspect.getfullargspec(func)
    max = len(argspec.args)
    if argspec.defaults is None:
        min = max
    else:
        min = max - len(argspec.defaults)
    if argspec.varargs is not None: # varargs means we can accept an unbounded #
        max = len(arguments)+1

    if not (min <= len(arguments) <= max):
        print_help(func, command)
        sys.exit(1)

    func(*arguments)

