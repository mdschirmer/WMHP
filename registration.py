#!/usr/bin/env python
"""
Registration module for WMH pipeline.

Handles ANTs-based registration to atlas space and inverse transforms.

Author: MDS
Organization: MGH/HMS
"""

from __future__ import annotations

__all__ = [
    'register_to_atlas',
    'apply_transform',
    'apply_inverse_transform',
    'RegistrationResult',
]

import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from subprocess import run, PIPE
from typing import Optional, Union, List

from utils import Log


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class RegistrationResult:
    """Container for registration outputs."""
    warped_image: str
    affine_transform: str
    transform_prefix: str
    success: bool = True
    error_message: Optional[str] = None


# =============================================================================
# ANTs Registration
# =============================================================================

def register_to_atlas(
    moving_image: Union[str, Path],
    fixed_image: Union[str, Path],
    output_prefix: Union[str, Path],
    transform_type: str = 'a',  # affine
    num_threads: int = 20,
    random_seed: int = 32,
    verbose: bool = False
) -> RegistrationResult:
    """
    Register moving image to atlas using ANTs.
    
    Parameters
    ----------
    moving_image
        Path to moving image (subject brain)
    fixed_image
        Path to fixed image (atlas)
    output_prefix
        Prefix for output transforms
    transform_type
        ANTs transform type: 'r' (rigid), 'a' (affine), 's' (syn)
    num_threads
        Number of CPU threads
    random_seed
        Random seed for reproducibility
    verbose
        Show ANTs output
        
    Returns
    -------
    RegistrationResult
        Container with transform paths and status
    """
    Log.step("Running ANTs registration...")
    
    # Construct command
    cmd = [
        'antsRegistrationSyNQuick.sh',
        '-d', '3',
        '-f', str(fixed_image),
        '-m', str(moving_image),
        '-t', transform_type,
        '-r', str(random_seed),
        '-n', str(num_threads),
        '-o', str(output_prefix)
    ]
    
    Log.debug(f"CMD: {' '.join(cmd)}")
    
    try:
        result = run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            Log.error(f"Registration failed (exit {result.returncode})")
            if result.stderr:
                Log.error(result.stderr)
            return RegistrationResult(
                warped_image='',
                affine_transform='',
                transform_prefix=str(output_prefix),
                success=False,
                error_message=result.stderr
            )
        
        if verbose and result.stdout:
            Log.info(result.stdout)
        
    except FileNotFoundError:
        msg = "antsRegistrationSyNQuick.sh not found. Please install ANTs."
        Log.error(msg)
        return RegistrationResult(
            warped_image='',
            affine_transform='',
            transform_prefix=str(output_prefix),
            success=False,
            error_message=msg
        )
    except Exception as e:
        Log.error(f"Registration error: {e}")
        return RegistrationResult(
            warped_image='',
            affine_transform='',
            transform_prefix=str(output_prefix),
            success=False,
            error_message=str(e)
        )
    
    # Check outputs
    affine = f"{output_prefix}0GenericAffine.mat"
    warped = f"{output_prefix}Warped.nii.gz"
    
    if not os.path.isfile(affine):
        msg = f"Registration produced no transform: {affine}"
        Log.error(msg)
        return RegistrationResult(
            warped_image='',
            affine_transform='',
            transform_prefix=str(output_prefix),
            success=False,
            error_message=msg
        )
    
    Log.ok("Registration complete")
    
    return RegistrationResult(
        warped_image=warped if os.path.isfile(warped) else '',
        affine_transform=affine,
        transform_prefix=str(output_prefix),
        success=True
    )


def apply_transform(
    input_image: Union[str, Path],
    output_image: Union[str, Path],
    reference_image: Union[str, Path],
    transforms: List[Union[str, Path]],
    interpolation: str = 'Linear',
    invert: bool = False
) -> bool:
    """
    Apply transforms to an image using WarpImageMultiTransform.
    
    Parameters
    ----------
    input_image
        Image to transform
    output_image
        Output path
    reference_image
        Reference space
    transforms
        List of transform files (applied in order)
    interpolation
        Interpolation method
    invert
        If True, invert the transforms
        
    Returns
    -------
    bool
        True if successful
    """
    cmd = [
        'WarpImageMultiTransform',
        '3',
        str(input_image),
        str(output_image),
        '-R', str(reference_image)
    ]
    
    # Add transforms
    for xfm in transforms:
        if invert and str(xfm).endswith('.mat'):
            cmd.extend(['-i', str(xfm)])
        else:
            cmd.append(str(xfm))
    
    Log.debug(f"CMD: {' '.join(cmd)}")
    
    try:
        result = run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            Log.error(f"Transform failed: {result.stderr}")
            return False
        
        return os.path.isfile(output_image)
        
    except FileNotFoundError:
        Log.error("WarpImageMultiTransform not found. Please install ANTs.")
        return False
    except Exception as e:
        Log.error(f"Transform error: {e}")
        return False


def apply_inverse_transform(
    input_image: Union[str, Path],
    output_image: Union[str, Path],
    reference_image: Union[str, Path],
    affine_transform: Union[str, Path]
) -> bool:
    """
    Apply inverse affine transform.
    
    Parameters
    ----------
    input_image
        Image to transform (in atlas space)
    output_image
        Output path (subject space)
    reference_image
        Reference image (subject space)
    affine_transform
        Affine transform file (.mat)
        
    Returns
    -------
    bool
        True if successful
    """
    Log.step("Applying inverse transform...")
    
    cmd = [
        'WarpImageMultiTransform',
        '3',
        str(input_image),
        str(output_image),
        '-R', str(reference_image),
        '-i', str(affine_transform)
    ]
    
    Log.debug(f"CMD: {' '.join(cmd)}")
    
    try:
        result = run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            Log.error(f"Inverse transform failed: {result.stderr}")
            return False
        
        if not os.path.isfile(output_image):
            Log.error(f"Inverse transform produced no output: {output_image}")
            return False
        
        Log.ok("Inverse transform complete")
        return True
        
    except FileNotFoundError:
        Log.error("WarpImageMultiTransform not found. Please install ANTs.")
        return False
    except Exception as e:
        Log.error(f"Inverse transform error: {e}")
        return False


def warp_to_atlas(
    input_image: Union[str, Path],
    output_image: Union[str, Path],
    reference_image: Union[str, Path],
    affine_transform: Union[str, Path]
) -> bool:
    """
    Warp image to atlas space using affine transform.
    
    Parameters
    ----------
    input_image
        Image to warp (subject space)
    output_image
        Output path (atlas space)
    reference_image
        Atlas image
    affine_transform
        Affine transform file (.mat)
        
    Returns
    -------
    bool
        True if successful
    """
    Log.step("Warping to atlas space...")
    
    return apply_transform(
        input_image=input_image,
        output_image=output_image,
        reference_image=reference_image,
        transforms=[affine_transform],
        invert=False
    )


def apply_legacy_atlas_transform(
    input_image: Union[str, Path],
    output_image: Union[str, Path],
    fixtures_dir: Union[str, Path],
    forward: bool = True
) -> bool:
    """
    Apply legacy atlas transform for nCerebro compatibility.
    
    The nCerebro model was trained on an older atlas. This applies the
    transform between the new atlas and the old one.
    
    Parameters
    ----------
    input_image
        Input image
    output_image
        Output path
    fixtures_dir
        Directory containing legacy transforms
    forward
        If True, new->old; if False, old->new
        
    Returns
    -------
    bool
        True if successful (or if transform not needed)
    """
    old_xfm = os.path.join(fixtures_dir, 'new_to_old0GenericAffine.mat')
    ref_img = os.path.join(fixtures_dir, 'iso_flair_template_intres_brain.nii.gz')
    
    if not os.path.isfile(old_xfm):
        Log.info("Legacy atlas transform not found, skipping")
        shutil.copy(str(input_image), str(output_image))
        return True
    
    if not os.path.isfile(ref_img):
        Log.warn(f"Legacy reference image not found: {ref_img}")
        shutil.copy(str(input_image), str(output_image))
        return True
    
    Log.debug(f"Applying legacy atlas transform (forward={forward})")
    
    return apply_transform(
        input_image=input_image,
        output_image=output_image,
        reference_image=ref_img,
        transforms=[old_xfm],
        invert=not forward
    )
