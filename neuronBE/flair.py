#!/usr/bin/env python
"""
:Summary:
	Deep learning based skull stripper for FLAIR images

:Description:
	Input: Filename to nifti image 
	Output: binary brain mask

	Run either from terminal:

	bin/skullstrip.sh path/to/subjects
	# needs adaptation to naming conventions

	or within python

	import skullstripper
	# skull strip; returns numpy array with data (not NIFTI!)
	out_img = skullstripper.strip(nii.get_data(), model=None)
	

:Requires:
	See requirements.txt

:TODO:

:AUTHOR: MDS
:ORGANIZATION: MGH/HMS
:CONTACT: software@markus-schirmer.com
:SINCE: 2017-07-23
:VERSION: 0.2
"""
#=============================================
# Metadata
#=============================================
__author__ = 'mds'
__contact__ = 'software@markus-schirmer.com'
__copyright__ = ''
__license__ = ''
__date__ = '2019-03'
__version__ = '0.3'

#=============================================
# Import statements
#=============================================
import sys
import getopt
import os
import nibabel as nib
import keras
import numpy as np
import scipy.ndimage as sn
import skimage.measure as skm
import keras.layers as KL

import pdb

import intres

#=============================================
# Helper functions
#=============================================

def get_model(file='trained/model.hdf5', loss='mse'):
	# needs a workaround as keras changed some things creating a lambda layer error when using user defined softmax alrong 3rd or 4th dimension
	ndims = 2
	keras_issue_workaround=os.path.join(os.path.dirname(file),'tmp_model_for_keras_bug.hdf5')

	# load model without the last layer
	tmp_model = keras.models.load_model(keras_issue_workaround, custom_objects={'loss':loss})

	# add the last layer
	kkl = KL.Lambda(lambda x: keras.activations.softmax(x, axis=ndims+1), name='seg_prediction_out')(tmp_model.outputs[0])

	# define the model
	model = keras.models.Model(tmp_model.inputs, kkl)
	# load model, either specified or our trained model
	# model = keras.models.load_model(file, custom_objects={'loss':loss})
	return model

def adjust_img_size(img, width, height):

	# check if padding or scaling is necessary
	curr_shape = img.shape
	# initialise values for doing nothing
	pad_w = 0
	pad_h = 0
	rescale_w = 1.
	rescale_h = 1.

	### check width
	if curr_shape[0]!=width:
		print('Width mismatch: %f vs required %f' %(curr_shape[0],width))
		if curr_shape[0]<width:
			pad_w = (width-curr_shape[0])/2.
		else:
			rescale_w = width/float(curr_shape[0])
	### check height
	if curr_shape[1]!=height:
		print('Height mismatch: %f vs required %f' %(curr_shape[1],height))
		if curr_shape[1]<height:
			pad_h = (height-curr_shape[1])/2.
		else:
			rescale_h = height/float(curr_shape[1])

	# pad image
	pad_size = ((int(np.floor(pad_w)), int(np.ceil(pad_w))),(int(np.floor(pad_h)), int(np.ceil(pad_h))),(0,0))
	img = np.pad(img, pad_size, 'constant' )

	# rescale image
	rescale_size = (rescale_w, rescale_h, 1.)
	img = sn.zoom(img, rescale_size)

	return img, pad_size, rescale_size

def revert_img_size(img, pad_size=None, rescale_size=None):

	# undo padding
	if pad_size is not None:
		## There must be a smarter way than this
		pad_w = pad_size[0]
		pad_h = pad_size[1]
		if pad_w[1] == 0:
			img = img[pad_w[0]:,:,:]
		else:
			img = img[pad_w[0]:-pad_w[1],:,:]
		if pad_h[1] == 0: 
			img = img[:,pad_h[0]:,:]
		else:
			img = img[:,pad_h[0]:-pad_h[1],:]
		if len(pad_size)==3:
			pad_z = pad_size[2]
			if pad_z[1] == 0:
				img = img[:,:,pad_z[0]:]
			else:
				img = img[:,:,pad_z[0]:-pad_z[1]]

	if rescale_size is not None:
		# undo scaling
		rescale_size = (1./rescale_size[0], 1./rescale_size[1], 1.)
		img = sn.zoom(img, rescale_size)

	return img

def adjust_intensities(img, percentile=97, clip_flag=True):

	# first rescale intensity by percentile
	max_int = np.percentile(img.flat, percentile).astype(float)
	img = np.multiply(img, 1/max_int)

	# clip intensity values between 0 and 1 
	# (i.e. if i<0 -> i=0 ;;; if i>1 -> i=1)
	if clip_flag:
		img = np.clip(img, 0, 1)

	return img

def pad_in_z(img):
	return np.pad(img, ((0,0),(0,0),(1,1)), 'constant')

def get_biggest_connected_component(img):
	# label connected components
	label_img = skm.label(img)

	# find biggest connected component (brain)
	volumes = []
	labels = []
	for label in np.unique(label_img[label_img!=0]):
		volumes.append(np.sum(label_img == label))
		labels.append(label)

	brain_label = labels[np.argmax(volumes)]

	return label_img==brain_label

def estimate_brain_mask(img):

	# get biggest connected component of foreground (assumed as value 1)
	brain_mask = get_biggest_connected_component(img)

	# close holes
	brain_mask = np.logical_not(get_biggest_connected_component(np.logical_not(brain_mask)))

	return brain_mask

def adjust_inplane_resolution(nii, target_inplane_resolution=(0.9,0.9), rescale_factor=None, target_dimensions=None):

	if rescale_factor is None:
		# get image and make sure it's float
		img = nii.get_fdata().astype(float)

		# get header
		header = nii.header
	
		# determine scaling factors
		x_scale = np.round(10*header['pixdim'][1]/target_inplane_resolution[0])/10.
		y_scale = np.round(10*header['pixdim'][2]/target_inplane_resolution[1])/10.
		z_scale = 1.
	else:
		# if rescale factor is given, assume nii is img
		# this is for undoing the scaling
		img = nii

		if target_dimensions is None:
			x_scale = 1./float(rescale_factor[0])
			y_scale = 1./float(rescale_factor[1])
			z_scale = 1./float(rescale_factor[2])
		else:
			x_scale = float(target_dimensions[0])/float(img.shape[0])
			y_scale = float(target_dimensions[1])/float(img.shape[1])
			z_scale = float(target_dimensions[2])/float(img.shape[2])

	inplane_resolution_scaling = (x_scale, y_scale, z_scale)

	# scale image
	# pdb.set_trace()
	img = sn.zoom(img, inplane_resolution_scaling)

	return img, inplane_resolution_scaling

def strip(nii, model=None, model_file=None, debug=False):#, training_size=[256.,256.]):
	# training size: legacy, as the initial model did not use padding
	# debug mode saves intermediate image after intensity rescaling (needs to be implemented)

	if model is None:
		if model_file is None:
			model = get_model()
		else:
			model = get_model(file=model_file)

	# get training set dimensions (specify as input) and number of slices
	# model_shape = model.input_shape
	# if training_size is None:
	# 	training_size = [0,0]
	# 	width = model_shape[1]
	# 	height = model_shape[2]
	# else:
	# 	width = training_size[0]
	# 	height = training_size[1]

	img, inplane_resolution_scaling = adjust_inplane_resolution(nii, target_inplane_resolution=(0.9,0.9), rescale_factor=None)

	# check image sizes
	# img, pad_size, rescale_size = adjust_img_size(img, width, height)

	# rescale intensities
	img = adjust_intensities(img, percentile=97, clip_flag=True)

	# additional padding to avoid convolutions to act up
	# assumed to be the difference between the model input shape and the training data shape
	# pad_for_processing = [(int(float(model_shape[1]-training_size[0])/2.),int(float(model_shape[1]-training_size[0])/2.)),(int(float(model_shape[2]-training_size[1])/2.),int(float(model_shape[2]-training_size[1])/2.)),(1,1)]
	# make sure it becomes multiple of 16
	pad_x_before = 16 + np.ceil(16-np.remainder(img.shape[0],16)/2.).astype(int)
	pad_x_after = 16 + np.floor(16-np.remainder(img.shape[0],16)/2.).astype(int)
	pad_y_before = 16 + np.ceil(16-np.remainder(img.shape[1],16)/2.).astype(int)
	pad_y_after = 16 + np.floor(16-np.remainder(img.shape[1],16)/2.).astype(int)
	pad_z = 1
	pad_for_processing = [(pad_x_before,pad_x_after),(pad_y_before,pad_y_after),(pad_z,pad_z)]
	img = np.pad(img, pad_for_processing, 'constant' )

	# get width, height and num slices
	num_slices = img.shape[2]
	width = img.shape[0]
	height = img.shape[1]

	# create output array for prediction
	# prob_map = np.zeros((num_slices, int(width*height), 2))
	prob_map = np.zeros((num_slices, int(width),int(height), 2))

	####################
	# old model
	####################

	# # img needs to be reordered to fit into the model (slice by slice stripping)
	# img = np.swapaxes(np.swapaxes(img, 1, 2), 0, 1)
	# # model needs the 4th dimension
	# img = np.expand_dims(img,3)
	# # predict brain mask slice-wise (requires 3 slices for prediction)
	# prob_map = np.squeeze(model.predict(img))

	####################
	# new model
	####################

	# predict brain mask slice-wise (requires 3 slices for prediction)
	for i_slice in np.arange(num_slices-2)+1:
		# get slice
		this_slice = img[:,:,i_slice-1:i_slice+2]
		# add extra dimension for input
		this_slice = np.expand_dims(this_slice,0)
		# predict
		prob_map[i_slice,:,:,:] = np.squeeze(model.predict(this_slice))

	# get label_map
	labels = np.argmax(prob_map, axis=3).astype(int)

	# reshape
	labels = labels.reshape(num_slices, int(width), int(height))
	labels = np.squeeze(np.swapaxes(np.swapaxes(labels, 0, 1), 1, 2))

	# revert padding for processing
	labels = revert_img_size(labels, pad_size=pad_for_processing)

	# return labels to previous shape
	# labels = revert_img_size(labels, pad_size=pad_size, rescale_size=rescale_size)

	# revert resolution change
	labels, dummy = adjust_inplane_resolution(labels, target_inplane_resolution=(0.9,0.9), rescale_factor=inplane_resolution_scaling, target_dimensions=nii.get_fdata().shape)

	# eliminate small connected components and holes
	# pdb.set_trace()
	brain_mask = estimate_brain_mask(labels)

	return brain_mask

def get_gmwm_mask(brain):
	# input needs to be intensity normalized
	# threshold, removing intensities below half of normal appearing WM
	img = (brain>0.375).astype(int)
	mask = np.zeros(img.shape)

	strct = np.stack((np.zeros((3,3)),sn.generate_binary_structure(2, 1), np.zeros((3,3))),2)

	img = sn.binary_erosion(img, structure=strct, iterations=1)
	img = get_biggest_connected_component(img)
	img = sn.binary_dilation(img, structure=strct, iterations=1)

	mask = np.logical_not(get_biggest_connected_component(np.logical_not(img)))
	for zz in np.arange(mask.shape[2]):
		if np.sum(mask[:,:,zz]>0):
			mask[:,:,zz] = sn.binary_fill_holes(mask[:,:,zz])

	return img, mask

def main(argv):
	###################################
	# catch input
	###################################
	try:
		opts, args = getopt.getopt(argv[1:],"hi:o:n:g:m:b:r:l:")
	except getopt.GetoptError:
		print('flair.py -i <inputfile> -o <output_brain_mask_file> (-b <output_brain_file> -n <output_intensity_normalized_file> -g <output_gmwm_mask_file> -m <model_file> -r <output_refined_brain_mask_file> -l <log_file>)')
		sys.exit(2)
		
	nii_file = None
	mask_file = None
	brain_file = None
	intres_file = None
	gmwm_file = None
	model_file = None
	updated_mask_file = None
	log_file = None

	for opt, arg in opts:
		if opt == '-h':
			print('flair.py -i <inputfile> -o <output_brain_mask_file> (-b <output_brain_file> -n <output_intensity_normalized_file> -g <output_gmwm_mask_file> -m <model_file> -r <output_refined_brain_mask_file> -l <log_file>)')
			sys.exit()
		elif opt in ("-i"):
			nii_file = arg
			if not os.path.isfile(nii_file):
				print('Input file %s not found.' %nii_file)
				sys.exit()
		elif opt in ("-o"):
			mask_file = arg
			if not os.path.isdir(os.path.dirname(mask_file)):
				print('Output directory %s does not exist.' %os.path.dirname(mask_file))
				sys.exit()
		elif opt in ("-b"):
			brain_file = arg
			if not os.path.isdir(os.path.dirname(brain_file)):
				print('Output directory %s does not exist.' %os.path.dirname(brain_file))
				sys.exit()
		elif opt in ("-n"):
			intres_file = arg
			if not os.path.isdir(os.path.dirname(intres_file)):
				print('Output directory %s does not exist.' %os.path.dirname(intres_file))
				sys.exit()
		elif opt in ("-g"):
			gmwm_file = arg
			if not os.path.isdir(os.path.dirname(gmwm_file)):
				print('Output directory %s does not exist.' %os.path.dirname(gmwm_file))
				sys.exit()
		elif opt in ("-m"):
			model_file = arg
			if not os.path.isfile(model_file):
				print('Model file %s not found. Reverting to initial model.' %model_file)
				model_file = None
		elif opt in ("-l"):
			log_file = arg
			if not os.path.isdir(os.path.dirname(log_file)):
				print('Output directory %s does not exist.' %os.path.dirname(log_file))
				sys.exit()
		elif opt in ("-r"):
			refine_mask = True
			updated_mask_file = arg
			if not os.path.isdir(os.path.dirname(updated_mask_file)):
				print('Output directory %s does not exist.' %os.path.dirname(updated_mask_file))
				sys.exit()

	if model_file is None:
		model_file = os.path.join(os.path.dirname(argv[0]), 'trained', 'model.hdf5')

	###################################
	###################################

	# load nifti image
	nii = nib.load(nii_file)
	# get header and adapt number type
	header = nii.header
	header['datatype'] = 16 # corresponds to float32
	header['bitpix'] = 32 # corresponds to float32
	# get affine 
	affine = nii.affine

	# skull strip
	mask = strip(nii, model=None, model_file = model_file)
	if mask_file is not None:
		# save image
		out_nii = nib.Nifti1Image(mask, header=header, affine=affine)
		out_nii.to_filename(mask_file)

	# prep brain image
	brain = np.multiply(nii.get_fdata().astype(float), mask.astype(float))
	brain[brain!=0] = brain[brain!=0] - np.min(brain)

	# intensity normalize
	intres_img, norm = intres.rescale(brain, mask=mask, logfile=log_file)

	if (gmwm_file is not None) or (updated_mask_file is not None):
		# get grey/white matter mask
		img, updated_mask = get_gmwm_mask(intres_img)
		if (gmwm_file is not None):
			out_nii = nib.Nifti1Image(img, header=header, affine=affine)
			out_nii.to_filename(gmwm_file)

		if (updated_mask_file is not None):
			mask = updated_mask
			# save image
			out_nii = nib.Nifti1Image(mask, header=header, affine=affine)
			out_nii.to_filename(updated_mask_file)

			# update brain image
			brain = np.multiply(nii.get_fdata().astype(float), mask.astype(float))
			brain[brain!=0] = brain[brain!=0] - np.min(brain)

			# update intres image
			intres_img = np.multiply(intres_img, mask)


	if intres_file is not None:
		out_nii = nib.Nifti1Image(intres_img, header=header, affine=affine)
		out_nii.to_filename(intres_file)

	if log_file is not None:
		# save rescaling value
		with open(log_file, 'a') as fid:
			fid.write('%s: %f \n' %('Rescaling factor', norm))

	if brain_file is not None:
		out_nii = nib.Nifti1Image(brain, header=header, affine=affine)
		out_nii.to_filename(brain_file)

if __name__ == "__main__":
	main(sys.argv)
