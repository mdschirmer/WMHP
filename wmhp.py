#!/usr/bin/env python3
"""
WMHP - White Matter Hyperintensity Pipeline

A streamlined pipeline for WMH segmentation from clinical FLAIR MRI.

Pipeline:
    1. Preprocessing (MR-CLOVER): Bias correction, brain extraction, normalization
    2. Registration: Register to atlas space (ANTs)
    3. WMH Segmentation: Run nCerebro model
    4. Post-processing: Inverse transform, binarize

Usage:
    ./wmhp.py -i flair.nii.gz -o wmh.nii.gz
    python wmhp.py -i flair.nii.gz -o wmh.nii.gz

Author: MDS
Organization: MGH/HMS
Contact: mschirmer1@mgh.harvard.edu
Version: 1.0.0
Date: 2025
"""

from __future__ import annotations

__version__ = '1.0.0'
__author__ = 'mds'

import argparse
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path
from subprocess import run
from typing import Optional

import nibabel as nib
import numpy as np

# Add script directory to path for local imports
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from preprocessing import (
    run_mr_clover,
    apply_bias_and_normalize,
    create_brain_matchwm,
    DEFAULT_NORM_TARGET,
)
from registration import (
    register_to_atlas,
    apply_transform,
    apply_inverse_transform,
    apply_legacy_atlas_transform,
)
from utils import (
    Log,
    binarize,
    write_stats_csv,
    get_wm_mode_from_stats,
)


# =============================================================================
# Configuration
# =============================================================================

def get_script_dir() -> str:
    """Get the wmhp directory."""
    return os.path.dirname(os.path.abspath(__file__))


def get_default_paths() -> dict:
    """
    Get default paths for pipeline components.
    
    Expects this structure:
        wmhp/
        ├── wmhp.py (this file)
        ├── nCerebro/
        │   ├── cerebro.py
        │   └── fixtures/
        └── atlas/
    """
    script_dir = get_script_dir()
    
    return {
        'ncerebro': os.path.join(script_dir, 'nCerebro', 'cerebro.py'),
        'atlas': os.path.join(script_dir, 'atlas', 'caa_flair_in_mni_template_smooth_brain_intres.nii.gz'),
        'fixtures': os.path.join(script_dir, 'nCerebro', 'fixtures'),
    }


# =============================================================================
# Helper Functions
# =============================================================================

def run_ncerebro(
    input_image: str,
    output_image: str,
    ncerebro_script: str,
    python_bin: str = 'python3'
) -> bool:
    """
    Run nCerebro WMH segmentation.
    
    Parameters
    ----------
    input_image
        Input image in atlas space
    output_image
        Output WMH probability map
    ncerebro_script
        Path to cerebro.py
    python_bin
        Python interpreter
        
    Returns
    -------
    bool
        True if successful
    """
    Log.step("Running WMH segmentation (nCerebro)...")
    
    cmd = [python_bin, '-u', ncerebro_script, input_image, output_image]
    Log.debug(f"CMD: {' '.join(cmd)}")
    
    try:
        result = run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            Log.error(f"nCerebro failed (exit {result.returncode})")
            if result.stderr:
                Log.error(result.stderr)
            return False
        
        if not os.path.isfile(output_image):
            Log.error("nCerebro produced no output")
            return False
        
        Log.ok("WMH segmentation complete")
        return True
        
    except FileNotFoundError:
        Log.error(f"Python not found: {python_bin}")
        return False
    except Exception as e:
        Log.error(f"nCerebro error: {e}")
        return False


# =============================================================================
# Main Pipeline
# =============================================================================

def process(args: argparse.Namespace) -> int:
    """
    Run the full WMH pipeline.
    
    Parameters
    ----------
    args
        Command-line arguments
        
    Returns
    -------
    int
        Exit code (0 for success)
    """
    start_time = time.time()
    
    # Configure logging
    Log.configure(verbose=args.verbose, debug=args.debug, quiet=args.quiet)
    
    # Get default paths
    defaults = get_default_paths()
    ncerebro_script = args.ncerebro or defaults['ncerebro']
    atlas_file = args.atlas or defaults['atlas']
    fixtures_dir = args.fixtures or defaults['fixtures']
    
    # Validate inputs
    if not os.path.isfile(args.input):
        Log.error(f"Input file not found: {args.input}")
        return 1
    
    if not os.path.isfile(ncerebro_script):
        Log.error(f"nCerebro script not found: {ncerebro_script}")
        return 1
    
    if not os.path.isfile(atlas_file):
        Log.error(f"Atlas file not found: {atlas_file}")
        return 1
    
    # Check for existing output
    if os.path.isfile(args.output) and not args.force:
        Log.ok(f"Output exists: {args.output}")
        return 0
    
    # Setup
    subject = args.subject or Path(args.input).stem.replace('.nii', '')
    work_dir = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(work_dir, exist_ok=True)
    
    tmpdir = tempfile.mkdtemp(prefix=f'wmhp_{subject}_', dir=work_dir)
    Log.debug(f"Temp directory: {tmpdir}")
    
    Log.step(f"Processing {subject}")
    Log.info(f"Input: {args.input}")
    Log.info(f"Output: {args.output}")
    
    try:
        # =====================================================================
        # Step 1: Preprocessing (MR-CLOVER)
        # =====================================================================
        
        # Define intermediate file paths
        brain_mask = os.path.join(tmpdir, f'{subject}_brain.nii.gz')
        norm_brain = os.path.join(tmpdir, f'{subject}_norm.nii.gz')
        gmwm_mask = os.path.join(tmpdir, f'{subject}_gmwm.nii.gz')
        bias_field = os.path.join(tmpdir, f'{subject}_bias.nii.gz')
        stats_file = os.path.join(tmpdir, f'{subject}_stats.csv')
        brain_matchwm = os.path.join(tmpdir, f'{subject}_matchwm.nii.gz')
        
        # Check for pre-computed inputs
        have_precomputed = (
            args.gmwm_mask and os.path.isfile(args.gmwm_mask) and
            args.bias_field and os.path.isfile(args.bias_field) and
            args.mr_clover_stats and os.path.isfile(args.mr_clover_stats)
        )
        
        if have_precomputed:
            Log.ok("Using pre-computed MR-CLOVER outputs")
            
            # Get WM mode from stats
            wm_mode = get_wm_mode_from_stats(args.mr_clover_stats)
            if wm_mode is None:
                Log.error(f"Could not read WM mode from: {args.mr_clover_stats}")
                return 1
            
            Log.info(f"WM mode: {wm_mode}")
            
            # Apply bias correction and normalization
            apply_bias_and_normalize(
                input_path=args.input,
                bias_field_path=args.bias_field,
                wm_mode=wm_mode,
                target=args.norm_target,
                output_path=norm_brain
            )
            
            # Use provided GMWM mask
            shutil.copy(args.gmwm_mask, gmwm_mask)
            brain_mask = gmwm_mask  # Use GMWM as brain mask
            
        else:
            # Run full preprocessing
            if not run_mr_clover(
                input_file=args.input,
                output_gmwm=gmwm_mask,
                output_brain=brain_mask,
                output_norm=norm_brain,
                output_bias_field=bias_field,
                output_stats=stats_file,
                subject_id=subject,
                norm_target=args.norm_target,
                use_cpu=args.cpu,
                robust=args.robust,
                force=args.force,
                verbose=args.verbose,
                python_bin=args.python or 'python3'
            ):
                Log.error("MR-CLOVER preprocessing failed")
                return 1
            
            if not os.path.isfile(gmwm_mask) or not os.path.isfile(norm_brain):
                Log.error("Preprocessing failed to produce required outputs")
                return 1
        
        # Create brain_matchwm
        if not create_brain_matchwm(norm_brain, gmwm_mask, brain_matchwm):
            return 1
        
        # Copy user-requested outputs
        if args.brain and not os.path.isfile(args.brain):
            shutil.copy(brain_mask, args.brain)
            Log.info(f"Saved: {args.brain}")
        
        if args.norm and not os.path.isfile(args.norm):
            shutil.copy(norm_brain, args.norm)
            Log.info(f"Saved: {args.norm}")
        
        if args.gmwm and not os.path.isfile(args.gmwm):
            shutil.copy(gmwm_mask, args.gmwm)
            Log.info(f"Saved: {args.gmwm}")
        
        # =====================================================================
        # Step 2: Registration
        # =====================================================================
        
        reg_dir = os.path.join(tmpdir, 'reg')
        os.makedirs(reg_dir, exist_ok=True)
        transform_prefix = os.path.join(reg_dir, f'{subject}_')
        atlas_reg = os.path.join(tmpdir, f'{subject}_atlas.nii.gz')
        orig_atlas = os.path.join(tmpdir, f'{subject}_orig_atlas.nii.gz')
        
        # Register to atlas
        reg_result = register_to_atlas(
            moving_image=brain_matchwm,
            fixed_image=atlas_file,
            output_prefix=transform_prefix,
            transform_type='a',
            num_threads=args.threads,
            verbose=args.verbose
        )
        
        if not reg_result.success:
            Log.error("Registration failed")
            return 1
        
        # Warp to atlas
        if not apply_transform(
            input_image=brain_matchwm,
            output_image=atlas_reg,
            reference_image=atlas_file,
            transforms=[reg_result.affine_transform]
        ):
            Log.error("Forward warp failed")
            return 1
        
        # Apply legacy atlas transform (for nCerebro compatibility)
        if not apply_legacy_atlas_transform(
            input_image=atlas_reg,
            output_image=orig_atlas,
            fixtures_dir=fixtures_dir,
            forward=True
        ):
            Log.error("Legacy transform failed")
            return 1
        
        # =====================================================================
        # Step 3: WMH Segmentation
        # =====================================================================
        
        wmh_atlas = os.path.join(tmpdir, f'{subject}_wmh_atlas.nii.gz')
        
        if not run_ncerebro(
            input_image=orig_atlas,
            output_image=wmh_atlas,
            ncerebro_script=ncerebro_script,
            python_bin=args.python or 'python3'
        ):
            return 1
        
        # =====================================================================
        # Step 4: Inverse Transform and Binarize
        # =====================================================================
        
        Log.step("Transforming to subject space...")
        
        wmh_tmp = os.path.join(tmpdir, f'{subject}_wmh_tmp.nii.gz')
        wmh_subj = os.path.join(tmpdir, f'{subject}_wmh_subj.nii.gz')
        
        # Inverse legacy transform
        if not apply_legacy_atlas_transform(
            input_image=wmh_atlas,
            output_image=wmh_tmp,
            fixtures_dir=fixtures_dir,
            forward=False
        ):
            Log.error("Inverse legacy transform failed")
            return 1
        
        # Inverse to subject space
        if not apply_inverse_transform(
            input_image=wmh_tmp,
            output_image=wmh_subj,
            reference_image=brain_matchwm,
            affine_transform=reg_result.affine_transform
        ):
            return 1
        
        # Binarize
        output_stats = args.stats or os.path.join(tmpdir, f'{subject}_wmh_stats.csv')
        
        binarize(
            input_path=wmh_subj,
            output_path=args.output,
            threshold=args.threshold,
            stats_path=output_stats,
            stats_name='WMH_volume_mm3',
            subject_id=subject
        )
        
        if not os.path.isfile(args.output):
            Log.error("No output produced")
            return 1
        
        # Copy stats if requested
        if args.stats and output_stats != args.stats:
            shutil.copy(output_stats, args.stats)
            Log.info(f"Saved: {args.stats}")
        
        # =====================================================================
        # Done
        # =====================================================================
        
        elapsed = time.time() - start_time
        elapsed_str = f"{int(elapsed // 60)}m{int(elapsed % 60)}s"
        Log.ok(f"Complete: {args.output} ({elapsed_str})")
        
        return 0
        
    except Exception as e:
        Log.error(f"Pipeline failed: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        return 1
        
    finally:
        if not args.keep:
            shutil.rmtree(tmpdir, ignore_errors=True)
        else:
            Log.debug(f"Keeping temp files: {tmpdir}")


# =============================================================================
# CLI
# =============================================================================

def main() -> int:
    """Command-line interface."""
    parser = argparse.ArgumentParser(
        description=f'WMHP v{__version__} - White Matter Hyperintensity Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Basic usage
  ./wmhp.py -i flair.nii.gz -o wmh.nii.gz
  
  # With outputs
  ./wmhp.py -i flair.nii.gz -o wmh.nii.gz --brain brain.nii.gz --stats volumes.csv
  
  # Using pre-computed MR-CLOVER outputs
  ./wmhp.py -i flair.nii.gz -o wmh.nii.gz \\
      --gmwm-mask gmwm.nii.gz --bias-field bias.nii.gz --mr-clover-stats stats.csv
'''
    )
    
    # Required
    parser.add_argument('-i', '--input', required=True,
                        help='Input FLAIR image')
    parser.add_argument('-o', '--output', required=True,
                        help='Output WMH mask')
    
    # Subject
    parser.add_argument('-s', '--subject',
                        help='Subject ID (default: from filename)')
    
    # Optional outputs
    output_group = parser.add_argument_group('Output files')
    output_group.add_argument('--stats',
                              help='Statistics CSV')
    output_group.add_argument('--brain',
                              help='Brain mask')
    output_group.add_argument('--norm',
                              help='Normalized brain')
    output_group.add_argument('--gmwm',
                              help='GM/WM segmentation')
    
    # Pre-computed inputs
    precomp_group = parser.add_argument_group('Pre-computed inputs (skip MR-CLOVER)')
    precomp_group.add_argument('--gmwm-mask',
                               help='Pre-computed GMWM mask')
    precomp_group.add_argument('--bias-field',
                               help='Pre-computed bias field')
    precomp_group.add_argument('--mr-clover-stats',
                               help='Pre-computed stats CSV (must contain wm_mode_intensity)')
    
    # Processing options
    proc_group = parser.add_argument_group('Processing options')
    proc_group.add_argument('-t', '--threads', type=int, default=20,
                            help='CPU threads (default: 20)')
    proc_group.add_argument('-T', '--threshold', type=float, default=0.5,
                            help='WMH threshold 0-1 (default: 0.5)')
    proc_group.add_argument('--norm-target', type=float, default=DEFAULT_NORM_TARGET,
                            help=f'Target WM intensity (default: {DEFAULT_NORM_TARGET})')
    proc_group.add_argument('--cpu', action='store_true',
                            help='Force CPU processing (default: GPU)')
    proc_group.add_argument('--robust', action='store_true',
                            help='Use SynthSeg robust mode')
    
    # Paths
    path_group = parser.add_argument_group('Custom paths')
    path_group.add_argument('--ncerebro',
                            help='Path to cerebro.py')
    path_group.add_argument('--atlas',
                            help='Path to atlas file')
    path_group.add_argument('--fixtures',
                            help='Path to nCerebro fixtures')
    path_group.add_argument('--python',
                            help='Python interpreter (default: python3)')
    
    # Flags
    flag_group = parser.add_argument_group('Flags')
    flag_group.add_argument('-f', '--force', action='store_true',
                            help='Force overwrite')
    flag_group.add_argument('-k', '--keep', action='store_true',
                            help='Keep temp files')
    flag_group.add_argument('-v', '--verbose', action='store_true',
                            help='Verbose output')
    flag_group.add_argument('-q', '--quiet', action='store_true',
                            help='Minimal output')
    flag_group.add_argument('--debug', action='store_true',
                            help='Debug mode')
    flag_group.add_argument('-V', '--version', action='version',
                            version=f'%(prog)s {__version__}')
    
    args = parser.parse_args()
    return process(args)


if __name__ == '__main__':
    sys.exit(main())
