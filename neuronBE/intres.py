#!/usr/bin/env python
"""
:Summary:

:Description:

:Requires:

:TODO:

:AUTHOR: MDS
:ORGANIZATION: MGH/HMS
:CONTACT: software@markus-schirmer.com
:SINCE: 2017-06-20
:VERSION: 0.1
"""
#=============================================
# Metadata
#=============================================
__author__ = 'mds'
__contact__ = 'software@markus-schirmer.com'
__copyright__ = ''
__license__ = ''
__date__ = '2017-06'
__version__ = '0.1'

#=============================================
# Import statements
#=============================================
import sys
import os
import numpy as np
import nibabel as nib
import warnings

# import matplotlib.pyplot as plt

#=============================================
# Helper functions
#=============================================

def mean_shift_mode_finder(data, sigma=None, n_replicates=10, replication_method='percentiles', epsilon=None, max_iterations=1000, n_bins=None):
	"""
	Finds the mode of data using mean shift. Returns the best
	value and its score.

	e.g., (mean, score) = mean_shift_mode_finder(data.flatten())

	Inputs
	------
	data : data to find the mode of (one-dimensional ndarray)
	sigma : kernel sigma (h) to be used; defaults to heuristic
	n_replicates : how many times to run
	replication_method : how to determine initialization for each replicate.
						   'percentile' (uses n_replicate percentiles)
						   'random' (uses n_replicate random valid values)
	epsilon : if the change in mode is less than this value, stop
	max_iterations : maximum number of iterations for each replicate
	n_bins : how many bins to use for the data histogram

	Adapted from 'meanShift.m' by Adrian Dalca (https://github.com/adalca/mgt/)
	and from 'advanced_tools.py' by Ramesh Sridharan (https://github.com/rameshvs/pyniftitools)
	"""

	if sigma is None:
		# Optimal bandwidth suggested by Bowman and Azzalini ('97) p31
		# adapted from ksr.m by Yi Cao
		sigma = np.median(np.abs(data-np.median(data))) / .6745 * (4./3./float(data.size))**0.2
	if epsilon is None:
		# heuristic
		epsilon = sigma / 100.
	if n_bins is None:
		n_bins = int(max(data.size / 10., 1))

	# Set up histogram
	dmin, dmax = data.min(), data.max()
	bins = np.linspace(dmin, dmax, n_bins)
	bin_size = (dmax - dmin) / (n_bins - 1.)
	(data_hist, _) = np.histogram(data, bins)
	bin_centers = bins[:-1] + .5 * bin_size

	# Set up replicates
	if replication_method == 'percentiles':
		if n_replicates > 1:
			percentiles = np.linspace(0, 100, n_replicates)
		else:
			percentiles = [50]

		inits = [np.percentile(data, p) for p in percentiles]

	elif replication_method == 'random':
		inits = np.random.uniform(data.min(), data.max(), n_replicates)

	scores = np.empty(n_replicates)
	means = np.empty(n_replicates)
	iter_counts = np.zeros(n_replicates) + np.inf
	# Core algorithm
	for i in range(n_replicates):
		mean = inits[i]
		change = np.inf
		for j in range(max_iterations):
			if change < epsilon:
				break
			weights = np.exp(-.5 * ((data - mean)/sigma) ** 2)
			assert weights.sum() != 0, "Weights sum to 0; increase sigma if appropriate (current val %f)" % sigma
			mean_old = mean
			mean = np.dot(weights, data) / float(weights.sum())
			change = np.abs(mean_old - mean)
			# print('%i, %f' %(j,change))

		if not j<(max_iterations-1):
			warnings.warn('Maximum number of iterations reached. %i' %max_iterations)
			print('Did not converge in replication %i/%i. Change: %f, Epsilon: %f, Iterations: %i' %(i+1, n_replicates, change, epsilon, max_iterations))

		means[i] = mean

		kernel = np.exp(-(bin_centers - mean)**2/(2*sigma**2))
		scores[i] = np.dot(kernel, data_hist)
		iter_counts[i] = j

	best = np.argmax(scores)
	n_good_replicates = np.sum(np.abs(means[best] - means) < sigma / 5.) - 1

	return (means[best], scores[best])

#=============================================
# Main method
#=============================================

def rescale(img, mask=None, new_intensity=0.75, mode=None):
	#############
	# normalise max to new_intensity
	#############
	# normalise to 0
	img = img.astype(np.float32)
	if mask is not None:
		assert mask.shape == img.shape, 'Mask is of different shape than image.'
	else:
		print('Estimating brain mask')
		prec=np.percentile(img, 5)
		mask = img>prec
	
	brain = np.multiply(img, mask)

	if mode=='percentile':
		# normalise 95%ile to 1000
		norm = np.mean(brain[brain > np.percentile(brain, 5)])
	else:
		# find mean shift mode
		(norm, score) = mean_shift_mode_finder(brain[brain>0.].flatten())

	# rescale intensity
	img = img * new_intensity/float(norm)

	return img, norm


def main(argv):

	if len(argv)>4:
		print('Error. (Implement error message)')
		sys.exit()

	infile = argv[1]
	outfile = argv[2]

	#############
	# check input and load data
	#############
	assert os.path.isfile(infile), "Input file %s not found." % infile
	nii = nib.load(infile)
	brain = nii.get_data()
	if np.any(brain<0):
		print("Some intensity values are negative. Please check input data (%s). Adjusting intensitise for now (+ min(intensities)). " %infile)
		brain = brain + np.min(brain)

	if len(argv)==4:
		maskfile = argv[3]
		if os.path.isfile(maskfile):
			mask = nib.load(maskfile).get_data()
		else:
			mask = None
	else:
		mask = None

	#############
	# rescale
	#############
	img, norm = rescale(brain, mask=mask)

	infofile=outfile.replace('nii.gz','')+'log'
	with open(infofile, 'r') as file:
		csv_reader = csv.DictReader(file)
		data = [row for row in csv_reader]
		if len(data)==0:
			data = {}
		else:
			data = data[0]
	# add new info
	with open(infofile, 'w') as fid:
		data["Rescaling_factor"] = norm
		writer = csv.DictWriter(fid, fieldnames=data.keys())
		writer.writeheader()
		writer.writerows([data])

	#############
	# save image
	#############
	header = nii.get_header()
	header['datatype'] = 16 # corresponds to float32
	header['bitpix'] = 32 # corresponds to float32
	out_nii = nib.Nifti1Image(img, affine=nii.get_affine(), header=header)
	out_nii.to_filename(outfile)


if __name__ == "__main__":
	main(sys.argv)
