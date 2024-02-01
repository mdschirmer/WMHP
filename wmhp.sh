#!/usr/bin/env bash
set -e

######################################
# Setup environment 
######################################

python_bin=python3
ants_warp_transform=/Packages/ANTs/build/ANTS-build/Examples/WarpImageMultiTransform
ants_registration_quick=/Packages/ANTs/ANTs/Scripts/antsRegistrationSyNQuick.sh

num_threads=4

# intended for debugging
keep_intermediates=true

# These environment variables should be all set if using the vanilla version of the algorithm
WMHP_folder=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
nCerebro_bin=${WMHP_folder}/nCerebro/cerebro.py
cerebro_fixtures_folder=${WMHP_folder}/nCerebro/fixtures
neuronBE_bin=${WMHP_folder}/neuronBE/flair.py
atlas=${WMHP_folder}/atlas/caa_flair_in_mni_template_smooth_brain_intres.nii.gz


######################################
# Actual script
# Modify below only if you know what you are doing
######################################

# define input image
raw_img=$1
# define output names and path
# get base img name to use for outputs
outfile=$2
out_base=$(basename ${outfile} | awk -F'.nii.gz' '{print $1}')
outdir=$(echo $outfile | awk -F'.nii.gz' '{print $1}')
echo ""
echo "Evaluating ${raw_img} with output in folder ${outdir}"
echo ""

# check if it the script has been run before by checking if folder exists
wmh_seg_file=${outfile}
if [ ! -d ${outdir} ]; then
	# check if file and folder structure exists
	if  [ ! "./${raw_img}" ] ; then
		echo "File ${raw_img} not found. Exiting."
		exit
	fi

	if [ ! -d ${outdir}/reg ]; then 
		echo "Creating folder structure."
		mkdir -p ${outdir}/reg
	fi

	#creates new folder for output files 
	echo "Creating folder structure."
	mkdir -p ${outdir};

else

	echo "WMH has been run previously. Check ${wmh_seg_file}."
	echo "Would you like to run again anyway [Y/N]?"
	read input
	str='Y'

	if [ "$input" == "$str" ]; then

		#deletes duplicated folder/files and recreates an empty folder for new files to go into from rerun
		rm -r ${outdir}
		echo "Creating folder structure."
		mkdir -p ${outdir};

		# check if file and folder structure exists
		if [ ! "./${raw_img}" ]; then
			echo "File ${raw_img} not found. Exiting."
			exit
		fi

		if [ ! -d ${outdir}/reg ]; then 
			echo "Creating folder structure."
			mkdir -p ${outdir}/reg
		fi

	else
		exit
	fi
fi

# runs script if directory doesn't exist or if it does exist and user enters "Y" to continue
if [[ ! -f ${outdir} || "$input" == "$str" ]]; then

	# Brain extraction with NeuronBE
	echo "Executing brain extraction."
	infile=${raw_img}
	outfile=${outdir}/${out_base}_brain_matchwm.nii.gz
	if [ ! -f ${outfile} ]; then
		cmd="${python_bin} -u ${neuronBE_bin} -i ${infile} -o ${outdir}/${out_base}_brainmask_02.nii.gz -b ${outdir}/${out_base}_brain.nii.gz -n ${outfile} -g ${outdir}/${out_base}_gmwm_seg.nii.gz -r ${outdir}/${out_base}_brainmask_01.nii.gz -l ${outdir}/${out_base}_stats.log"
		echo ${cmd}; eval ${cmd};

		# extract gmwm volume info
		cmd="${python_bin} ${WMHP_folder}/tools.py binarize ${outdir}/${out_base}_gmwm_seg.nii.gz None 0.5 ${outdir}/${out_base}_stats.log GMWM_volume"
		echo ${cmd}; eval ${cmd};

	else
		echo "Brain extraction has been run previously. Using ${outfile}."
	fi

	# Register label-blurred flair atlas to subject
	echo "Running registration."
	infile=${outfile}
	outfile=${outdir}/${out_base}_in_orig_atlas.nii.gz
	if [ ! -f ${outfile} ]; then
		cmd="${ants_registration_quick} -d 3 -f ${atlas} -m ${infile} -t a -r 32 -n ${num_threads} -o ${outdir}/reg/${out_base}_brain_matchwm_in_atlas_"
		echo ${cmd}; eval ${cmd};

		# Warp subject image to atlas space using warp
		cmd="${ants_warp_transform} 3 ${infile} ${outdir}/${out_base}_in_atlas.nii.gz -R ${atlas} ${outdir}/reg/${out_base}_brain_matchwm_in_atlas_0GenericAffine.mat"
		echo ${cmd}; eval ${cmd};

		# Warp subject in atlas image to original atlas space in which nCerebro was trained
		cmd="${ants_warp_transform} 3 ${outdir}/${out_base}_in_atlas.nii.gz ${outdir}/${out_base}_in_orig_atlas.nii.gz -R ${cerebro_fixtures_folder}/iso_flair_template_intres_brain.nii.gz ${cerebro_fixtures_folder}/new_to_old0GenericAffine.mat"
		echo ${cmd}; eval ${cmd};

	else
		echo "Registration has been run previously. Using ${outfile}."
	fi

	# Segment WMH with cerebro
	echo "Running WMH segmentation."
	infile=${outfile}
	outfile=${wmh_seg_file}

	cmd="${python_bin} -u ${nCerebro_bin} ${infile} ${outdir}/${out_base}_wmh_in_orig_atlas.nii.gz"
	echo ${cmd}; eval ${cmd};
	
	# Warp subject in orig atlas image to atlas space using warp
	cmd="${ants_warp_transform} 3 ${outdir}/${out_base}_wmh_in_orig_atlas.nii.gz ${outdir}/${out_base}_wmh_in_atlas.nii.gz -R ${outdir}/${out_base}_in_atlas.nii.gz -i ${cerebro_fixtures_folder}/new_to_old0GenericAffine.mat"
	echo ${cmd}; eval ${cmd};
	
	# Warp wmh seg to upsampled img space using warp
	cmd="${ants_warp_transform} 3 ${outdir}/${out_base}_wmh_in_atlas.nii.gz ${outdir}/${out_base}_wmh_in_subject.nii.gz -R ${outdir}/${out_base}_brain_matchwm.nii.gz -i ${outdir}/reg/${out_base}_brain_matchwm_in_atlas_0GenericAffine.mat"
	echo ${cmd}; eval ${cmd};
	
	# Binarize image
	cmd="${python_bin} ${WMHP_folder}/tools.py binarize ${outdir}/${out_base}_wmh_in_subject.nii.gz ${outfile} 0.5 ${outdir}/${out_base}_stats.log WMH_volume"
	echo ${cmd}; eval ${cmd};
fi

# Cleaning up if requested
if ! ${keep_intermediates}; then
	# handle brain extraction files
	if [[ -f ${outdir}/${out_base}_brain.nii.gz ]]; then 
		rm ${outdir}/${out_base}_brain.nii.gz 
	fi
	if [[ -f ${outdir}/${out_base}_brain_matchwm.nii.gz ]]; then 
		rm ${outdir}/${out_base}_brain_matchwm.nii.gz
	fi 
	if [[ -f ${outdir}/${out_base}_brainmask_02.nii.gz ]]; then
		rm ${outdir}/${out_base}_brainmask_02.nii.gz
	fi

	# handle registration files
	if [[ -d ${outdir}/reg ]]; then 
		rm -r ${outdir}/reg
	fi
	if [[ -f ${outdir}/${out_base}_in_atlas.nii.gz ]]; then 
		rm ${outdir}/${out_base}_in_atlas.nii.gz 
	fi
	if [[ -f ${outdir}/${out_base}_in_orig_atlas.nii.gz ]]; then
		rm ${outdir}/${out_base}_in_orig_atlas.nii.gz
	fi

	# handle cerebro files
	if [[ -f ${outdir}/${out_base}_wmh_in_orig_atlas.nii.gz ]]; then 
	rm ${outdir}/${out_base}_wmh_in_orig_atlas.nii.gz 
	fi 
	if [[ -f ${outdir}/${out_base}_wmh_in_atlas.nii.gz ]]; then
		rm ${outdir}/${out_base}_wmh_in_atlas.nii.gz
	fi
	if [[ -f ${outdir}/${out_base}_wmh_in_subject.nii.gz ]]; then 
		rm ${outdir}/${out_base}_wmh_in_subject.nii.gz 
	fi
fi

echo "-----"
echo "Done. To look at results use e.g."
echo "fsleyes ${raw_img} ${wmh_seg_file}"
