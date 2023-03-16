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
__date__ = '2017-07'
__version__ = '0.2'

#=============================================
# Import statements
#=============================================
import sys
import os
import nibabel as nib
import keras
import numpy as np
import scipy.ndimage.interpolation as sni
import skimage.measure as skm
import keras.layers as KL
import keras.backend as K

import pdb

#=============================================
# Helper functions
#=============================================

def get_model(file='trained/model.hdf5', loss='mse'):
	
	seg_file = os.path.join(os.path.dirname(file),'tmp_seg_model_for_keras_bug.hdf5')
	seg_model = keras.models.load_model(seg_file, custom_objects={'loss':loss})

	ae_file = os.path.join(os.path.dirname(file),'tmp_ae_model_for_keras_bug.hdf5')
	ae_model = keras.models.load_model(ae_file, custom_objects={'loss':loss})

	# take seg model and create softmax-log layers
	softmax_lambda_fcn = lambda x: keras.activations.softmax(x, axis=4)
	seg_pred_output = KL.Lambda(softmax_lambda_fcn, name='seg_prediction')(seg_model.output)
	log_layer = KL.Lambda(lambda x: K.log(x + K.epsilon()), name='seg-log')
	seg_log_out = log_layer(seg_pred_output)

	# run through AE
	complex_log_ae_out = ae_model(seg_log_out)

	# need to softmax-then-log to make sure the prior is normalized.
	tmp_layer = KL.Lambda(lambda x: keras.activations.softmax(x, axis=4), name='seg-seg-softmax')
	log_layer = KL.Lambda(lambda x: K.log(x + K.epsilon()), name='seg-seg-soft-max-log')
	complex_ae_out = log_layer(tmp_layer(complex_log_ae_out))

	# join seg with AE
	k = keras.layers.Add(name='final_log_add')([complex_ae_out, seg_log_out])
	out_layer = KL.Lambda(lambda x: keras.activations.softmax(x, axis=4), name='seg-softmax-act')
	out_tensor = out_layer(k)

	# final model
	model = keras.models.Model(seg_model.inputs, out_tensor)

	return model

def get_model_legacy(file='trained/model.hdf5', loss='mse'):
	# needs a workaround as keras changed some things creating a lambda layer error when using user defined softmax alrong 3rd or 4th dimension
	ndims = 3
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

def create_mask(img, cut=[(32,16),(71,73),(20,12)]):

	# initialise
	mask = np.zeros(img.shape)

	# fill with 1s
	mask[cut[0][0]:-cut[0][1],cut[1][0]:-cut[1][1],cut[2][0]:-cut[2][1]] = 1

	# cut the image
	cropped_img = img[cut[0][0]:-cut[0][1],cut[1][0]:-cut[1][1],cut[2][0]:-cut[2][1]]

	return cropped_img, mask
	

def segment_WMH(nii, model=None, model_file=None, debug=False, legacy=True):

	# load model
	# if model is None:
	# 	model = get_model_legacy()

	if legacy:
		model = get_model_legacy(file=model_file)
	else:
		model = get_model(file=model_file)


	# create masks/crop image
	cropped_img, mask = create_mask(nii.get_fdata()) 
	#nonlinear cut=[(53, 54), (75, 80), (24, 24)])

	# add 4th dimension
	cropped_img = np.expand_dims(cropped_img,0)
	cropped_img = np.expand_dims(cropped_img,4)

	# segment WMH
	segmentation = np.squeeze(model.predict(cropped_img))
	
	# fill in segmentation 
	mask[mask==1] = np.ravel(segmentation[:,:,:,1])

	# threshold
	#mask[mask<0.5] = 0
	#mask[mask!=0] = 1

	return mask

def main(argv):

	# catch input 
	nii_file = argv[1]
	out_file = argv[2]

	if len(argv) == 4:
		model_file = argv[3]
	else:
		model_file = os.path.join(os.path.dirname(argv[0]), 'trained', 'model.hdf5')
	if not os.path.isfile(model_file):
		model_file = None

	# load nifti image
	nii = nib.load(nii_file)

	# segment WMH
	out_img = segment_WMH(nii, model=None, model_file = model_file, legacy=True)

	# save image
	out_nii = nib.Nifti1Image(out_img, header=nii.header, affine=nii.affine)
	out_nii.to_filename(out_file)


if __name__ == "__main__":
	main(sys.argv)
