#!/usr/bin/env python
"""
Preprocessing utilities for WMH pipeline.

This module provides helper functions for:
- Applying pre-computed bias fields and normalization
- Reading/writing statistics
- Mask operations

The main preprocessing is handled by mr_clover.py (called as subprocess).

Author: MDS
Organization: MGH/HMS
"""

from __future__ import annotations

__all__ = [
    'apply_bias_and_normalize',
    'apply_bias_correction',
    'create_brain_matchwm',
    'run_mr_clover',
    'DEFAULT_NORM_TARGET',
]

import os
from pathlib import Path
from subprocess import run
from typing import Optional, Union

import nibabel as nib
import numpy as np

from utils import Log, get_wm_mode_from_stats


# =============================================================================
# Constants
# =============================================================================

DEFAULT_NORM_TARGET = 0.75


# =============================================================================
# MR-CLOVER Wrapper
# =============================================================================

def get_mr_clover_path() -> str:
    """Get path to mr_clover.py script."""
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), 'mr_clover.py')


def run_mr_clover(
    input_file: Union[str, Path],
    output_gmwm: Optional[Union[str, Path]] = None,
    output_brain: Optional[Union[str, Path]] = None,
    output_norm: Optional[Union[str, Path]] = None,
    output_bias_field: Optional[Union[str, Path]] = None,
    output_stats: Optional[Union[str, Path]] = None,
    subject_id: Optional[str] = None,
    norm_target: float = DEFAULT_NORM_TARGET,
    use_cpu: bool = False,
    robust: bool = False,
    force: bool = False,
    verbose: bool = False,
    python_bin: str = 'python3'
) -> bool:
    """
    Run MR-CLOVER preprocessing pipeline.
    
    Parameters
    ----------
    input_file
        Input NIfTI image
    output_gmwm
        Output GM/WM mask
    output_brain
        Output brain mask
    output_norm
        Output normalized image
    output_bias_field
        Output bias field
    output_stats
        Output statistics CSV
    subject_id
        Subject identifier
    norm_target
        Target WM intensity for normalization
    use_cpu
        Force CPU processing
    robust
        Use SynthSeg robust mode
    force
        Force reprocessing
    verbose
        Show verbose output
    python_bin
        Python interpreter
        
    Returns
    -------
    bool
        True if successful
    """
    Log.step("Running MR-CLOVER preprocessing...")
    
    mr_clover_script = get_mr_clover_path()
    
    if not os.path.isfile(mr_clover_script):
        Log.error(f"mr_clover.py not found: {mr_clover_script}")
        return False
    
    # Build command
    cmd = [python_bin, mr_clover_script, '-i', str(input_file)]
    
    if output_gmwm:
        cmd.extend(['-o', str(output_gmwm)])
    if output_brain:
        cmd.extend(['--brain', str(output_brain)])
    if output_norm:
        cmd.extend(['--norm', str(output_norm)])
    if output_bias_field:
        cmd.extend(['--bias-field', str(output_bias_field)])
    if output_stats:
        cmd.extend(['--stats', str(output_stats)])
    if subject_id:
        cmd.extend(['--subject', subject_id])
    
    cmd.extend(['--norm-target', str(norm_target)])
    
    if use_cpu:
        cmd.append('--cpu')
    if robust:
        cmd.append('--robust')
    if force:
        cmd.append('--force')
    if verbose:
        cmd.append('--verbose')
    
    Log.debug(f"CMD: {' '.join(cmd)}")
    
    try:
        result = run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            Log.error(f"MR-CLOVER failed (exit {result.returncode})")
            if result.stderr:
                Log.error(result.stderr)
            return False
        
        if verbose and result.stdout:
            print(result.stdout)
        
        Log.ok("MR-CLOVER complete")
        return True
        
    except FileNotFoundError:
        Log.error(f"Python not found: {python_bin}")
        return False
    except Exception as e:
        Log.error(f"MR-CLOVER error: {e}")
        return False


# =============================================================================
# Bias Field and Normalization
# =============================================================================

def apply_bias_correction(
    input_path: Union[str, Path],
    bias_field_path: Union[str, Path],
    output_path: Optional[Union[str, Path]] = None
) -> nib.Nifti1Image:
    """
    Apply pre-computed bias field correction.
    
    Corrected = Input / BiasField
    
    Parameters
    ----------
    input_path
        Path to input image
    bias_field_path
        Path to bias field
    output_path
        Optional path to save result
        
    Returns
    -------
    nib.Nifti1Image
        Bias-corrected image
    """
    img = nib.load(input_path)
    bias = nib.load(bias_field_path)
    
    img_data = img.get_fdata().astype(np.float64)
    bias_data = bias.get_fdata().astype(np.float64)
    bias_data = np.where(bias_data > 0, bias_data, 1.0)
    
    corrected_data = img_data / bias_data
    corrected = nib.Nifti1Image(corrected_data.astype(np.float32), img.affine, img.header)
    
    if output_path:
        nib.save(corrected, str(output_path))
        Log.debug(f"Saved bias-corrected image: {output_path}")
    
    return corrected


def apply_bias_and_normalize(
    input_path: Union[str, Path],
    bias_field_path: Union[str, Path],
    wm_mode: float,
    target: float = DEFAULT_NORM_TARGET,
    output_path: Optional[Union[str, Path]] = None
) -> nib.Nifti1Image:
    """
    Apply bias field correction and WM intensity normalization.
    
    Parameters
    ----------
    input_path
        Path to raw input image
    bias_field_path
        Path to bias field
    wm_mode
        WM mode intensity (from MR-CLOVER stats)
    target
        Target intensity for WM
    output_path
        Optional path to save result
        
    Returns
    -------
    nib.Nifti1Image
        Bias-corrected and normalized image
    """
    Log.info(f"Applying bias correction (wm={wm_mode:.2f}, target={target})")
    
    if wm_mode <= 0:
        raise ValueError(f"Invalid WM mode value: {wm_mode}")
    
    # Apply bias correction
    corrected = apply_bias_correction(input_path, bias_field_path)
    
    # Normalize
    factor = target / wm_mode
    corrected_data = corrected.get_fdata()
    normalized_data = corrected_data * factor
    
    normalized = nib.Nifti1Image(
        normalized_data.astype(np.float32),
        corrected.affine,
        corrected.header
    )
    
    if output_path:
        nib.save(normalized, str(output_path))
        Log.ok(f"Saved normalized image: {output_path}")
    
    return normalized


def create_brain_matchwm(
    norm_image: Union[str, Path],
    mask_image: Union[str, Path],
    output_path: Union[str, Path]
) -> bool:
    """
    Create brain image masked and matched to WM intensity.
    
    Parameters
    ----------
    norm_image
        Normalized brain image
    mask_image
        Brain or GMWM mask
    output_path
        Output path
        
    Returns
    -------
    bool
        True if successful
    """
    Log.debug("Creating brain_matchwm")
    
    try:
        norm_nii = nib.load(norm_image)
        mask_nii = nib.load(mask_image)
        
        norm_data = norm_nii.get_fdata().astype(np.float32)
        mask_data = (mask_nii.get_fdata() > 0).astype(np.float32)
        
        out_data = norm_data * mask_data
        out_nii = nib.Nifti1Image(out_data, norm_nii.affine, norm_nii.header)
        nib.save(out_nii, str(output_path))
        
        Log.debug(f"Created: {output_path}")
        return True
        
    except Exception as e:
        Log.error(f"Failed to create brain_matchwm: {e}")
        return False
