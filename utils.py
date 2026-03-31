#!/usr/bin/env python
"""
Utility functions for WMH pipeline.

Includes:
- Logging with color-coded output
- NIfTI helpers (binarize, upsample, downsample)
- Statistics I/O

Author: MDS
Organization: MGH/HMS
"""

from __future__ import annotations

__all__ = [
    'Log',
    'binarize',
    'upsample',
    'downsample',
    'read_stats_csv',
    'write_stats_csv',
    'get_wm_mode_from_stats',
]

import csv
import os
import pickle
import sys
from pathlib import Path
from typing import Optional, Union

import nibabel as nib
import numpy as np
import scipy.ndimage as sn
from numpy.typing import NDArray


# =============================================================================
# Logging
# =============================================================================

class Log:
    """Simple color-coded logging."""
    
    COLORS = {
        'error': '\033[91m',
        'warn': '\033[93m',
        'ok': '\033[92m',
        'info': '\033[94m',
        'debug': '\033[95m',
        'step': '\033[96m',
        'reset': '\033[0m'
    }
    
    verbose: bool = False
    debug_mode: bool = False
    quiet: bool = False
    
    @classmethod
    def configure(cls, verbose: bool = False, debug: bool = False, quiet: bool = False) -> None:
        """Configure logging levels."""
        cls.verbose = verbose
        cls.debug_mode = debug
        cls.quiet = quiet
    
    @classmethod
    def error(cls, msg: str) -> None:
        print(f"{cls.COLORS['error']}[ERROR] {msg}{cls.COLORS['reset']}", file=sys.stderr)
    
    @classmethod
    def warn(cls, msg: str) -> None:
        print(f"{cls.COLORS['warn']}[WARN] {msg}{cls.COLORS['reset']}", file=sys.stderr)
    
    @classmethod
    def ok(cls, msg: str) -> None:
        if not cls.quiet:
            print(f"{cls.COLORS['ok']}[OK] {msg}{cls.COLORS['reset']}")
    
    @classmethod
    def step(cls, msg: str) -> None:
        if not cls.quiet:
            print(f"{cls.COLORS['step']}[STEP] {msg}{cls.COLORS['reset']}")
    
    @classmethod
    def info(cls, msg: str) -> None:
        if cls.verbose:
            print(f"{cls.COLORS['info']}[INFO] {msg}{cls.COLORS['reset']}")
    
    @classmethod
    def debug(cls, msg: str) -> None:
        if cls.debug_mode:
            print(f"{cls.COLORS['debug']}[DEBUG] {msg}{cls.COLORS['reset']}")


# =============================================================================
# Statistics I/O
# =============================================================================

def read_stats_csv(path: Union[str, Path]) -> dict[str, str]:
    """Read stats CSV and return as dict."""
    with open(path) as f:
        reader = csv.DictReader(f)
        row = next(reader, {})
        return dict(row)


def write_stats_csv(
    stats: Union[list[tuple[str, str]], dict[str, str]],
    path: Union[str, Path],
    append: bool = False
) -> None:
    """
    Write statistics to CSV file.
    
    Parameters
    ----------
    stats
        Either list of (key, value) tuples or a dict
    path
        Output CSV path
    append
        If True and file exists, merge with existing data
    """
    # Convert to dict if needed
    if isinstance(stats, list):
        data = {k: v for k, v in stats}
    else:
        data = dict(stats)
    
    # Merge with existing if appending
    if append and os.path.exists(path):
        try:
            existing = read_stats_csv(path)
            existing.update(data)
            data = existing
        except (IOError, csv.Error):
            pass
    
    # Write
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=data.keys())
        writer.writeheader()
        writer.writerow(data)


def get_wm_mode_from_stats(path: Union[str, Path]) -> Optional[float]:
    """
    Extract WM mode intensity from stats CSV.
    
    Returns
    -------
    float or None
        WM mode value, or None if not found
    """
    try:
        stats = read_stats_csv(path)
        
        for key in ['wm_mode_intensity', 'tissue_mode_intensity']:
            if key in stats and stats[key]:
                value = float(stats[key])
                if value > 0:
                    return value
    except (IOError, ValueError, KeyError):
        pass
    
    return None


# =============================================================================
# NIfTI Operations
# =============================================================================

def binarize(
    input_path: Union[str, Path],
    output_path: Optional[Union[str, Path]] = None,
    threshold: float = 0.5,
    stats_path: Optional[Union[str, Path]] = None,
    stats_name: Optional[str] = None,
    subject_id: Optional[str] = None
) -> nib.Nifti1Image:
    """
    Binarize a NIfTI image based on a threshold.
    
    Parameters
    ----------
    input_path
        Input NIfTI file
    output_path
        Output NIfTI file (None to skip saving)
    threshold
        Threshold value for binarization
    stats_path
        Optional CSV file to save volume statistics
    stats_name
        Column name for the volume statistic
    subject_id
        Subject ID for statistics
        
    Returns
    -------
    nib.Nifti1Image
        Binarized image
    """
    nii = nib.load(input_path)
    vol = nii.get_fdata()
    binary = (vol >= threshold).astype(np.uint8)
    
    # Create output image
    out_nii = nib.Nifti1Image(binary, affine=nii.affine, header=nii.header)
    
    # Save if requested
    if output_path is not None:
        out_nii.to_filename(str(output_path))
        Log.debug(f"Saved binarized image: {output_path}")
    
    # Save statistics if requested
    if stats_path is not None and stats_name is not None:
        voxel_size = float(np.prod(np.array(nii.header.get_zooms()[:3])))
        volume_mm3 = float(np.sum(binary)) * voxel_size
        
        stats = {}
        if subject_id:
            stats['subject'] = subject_id
        stats[stats_name] = f'{volume_mm3:.2f}'
        
        write_stats_csv(stats, stats_path, append=True)
        Log.debug(f"Saved volume stats: {stats_name}={volume_mm3:.2f} mm³")
    
    return out_nii


def upsample(
    input_path: Union[str, Path],
    output_path: Union[str, Path],
    zoom_log_path: Union[str, Path] = 'upsampling_log.pickle',
    isotropic: bool = True,
    factor: Optional[float] = None,
    order: int = 3
) -> nib.Nifti1Image:
    """
    Upsample a NIfTI image, saving zoom factors for later downsampling.
    
    Parameters
    ----------
    input_path
        Input NIfTI file
    output_path
        Output NIfTI file
    zoom_log_path
        Path to save zoom factors for reversal
    isotropic
        If True, upsample to isotropic resolution
    factor
        Fixed upsampling factor (overrides isotropic)
    order
        Spline interpolation order
        
    Returns
    -------
    nib.Nifti1Image
        Upsampled image
    """
    nii = nib.load(input_path)
    header = nii.header.copy()
    affine = nii.affine.copy()
    
    # Set output dtype
    header['datatype'] = 16  # float32
    header['bitpix'] = 32
    
    # Skip if no upsampling requested
    if not isotropic and factor is None:
        Log.info('Upsampling not requested, skipping...')
        nii.to_filename(str(output_path))
        return nii
    
    # Calculate zoom factors
    zooms = np.array(header.get_zooms()[:3])
    
    if isotropic:
        target_res = float(np.min(zooms))
        zoom_factors = [float(z) / target_res for z in zooms]
        zoom_factors = [max(1.0, round(z)) for z in zoom_factors]
    else:
        zoom_factors = [factor] * 3
    
    # Get old shape and data
    old_data = np.squeeze(nii.get_fdata().astype(np.float32))
    old_shape = old_data.shape
    
    Log.info(f'Upsampling with factors: {zoom_factors}')
    Log.debug(f'Old shape: {old_shape}')
    
    # Upsample
    new_data = sn.zoom(old_data, zoom_factors, order=order)
    Log.debug(f'New shape: {new_data.shape}')
    
    # Update header zooms
    old_zooms = list(header.get_zooms())
    new_zooms = [old_zooms[i] / zoom_factors[i] for i in range(3)]
    if len(old_zooms) > 3:
        new_zooms.extend(old_zooms[3:])
    header.set_zooms(new_zooms)
    
    # Update affine
    scaling = np.diag([1.0 / z for z in zoom_factors] + [1.0])
    affine = np.dot(affine, scaling)
    
    # Create and save output
    out_nii = nib.Nifti1Image(new_data, affine=affine, header=header)
    out_nii.to_filename(str(output_path))
    
    # Save zoom factors for downsampling
    with open(zoom_log_path, 'wb') as f:
        pickle.dump([zoom_factors, order, old_shape], f)
    
    Log.ok(f'Upsampled image saved: {output_path}')
    return out_nii


def downsample(
    input_path: Union[str, Path],
    output_path: Union[str, Path],
    zoom_log_path: Union[str, Path] = 'upsampling_log.pickle',
    order: int = 3
) -> nib.Nifti1Image:
    """
    Downsample a NIfTI image using saved zoom factors.
    
    Parameters
    ----------
    input_path
        Input NIfTI file (upsampled)
    output_path
        Output NIfTI file
    zoom_log_path
        Path to saved zoom factors
    order
        Spline interpolation order
        
    Returns
    -------
    nib.Nifti1Image
        Downsampled image
    """
    nii = nib.load(input_path)
    header = nii.header.copy()
    
    # Set output dtype
    header['datatype'] = 16  # float32
    header['bitpix'] = 32
    
    # Load zoom factors
    with open(zoom_log_path, 'rb') as f:
        zoom_factors, _, old_shape = pickle.load(f)
    
    # Calculate downsample factors
    current_shape = nii.get_fdata().shape
    downsample_factors = [
        float(old_shape[i]) / float(current_shape[i])
        for i in range(len(old_shape))
    ]
    
    Log.info(f'Downsampling with factors: {downsample_factors}')
    
    # Downsample
    new_data = sn.zoom(nii.get_fdata(), downsample_factors, order=order)
    
    # Update header zooms
    old_zooms = list(header.get_zooms())
    new_zooms = [old_zooms[i] / downsample_factors[i] for i in range(3)]
    if len(old_zooms) > 3:
        new_zooms.extend(old_zooms[3:])
    header.set_zooms(new_zooms)
    
    # Update affine
    downsample_factors_4d = list(downsample_factors) + [1.0]
    scaling = np.diag([1.0 / d for d in downsample_factors_4d])
    affine = np.dot(nii.affine, scaling)
    
    # Create and save output
    out_nii = nib.Nifti1Image(new_data.astype(np.float32), affine=affine, header=header)
    out_nii.to_filename(str(output_path))
    
    Log.ok(f'Downsampled image saved: {output_path}')
    return out_nii


# =============================================================================
# CLI for standalone usage
# =============================================================================

def _cli_main() -> int:
    """Command-line interface for standalone usage."""
    import inspect
    
    functions = {
        'binarize': binarize,
        'upsample': upsample,
        'downsample': downsample,
    }
    
    def print_help(func, name):
        spec = inspect.getfullargspec(func)
        args = ' '.join(spec.args)
        print(f'{sys.argv[0]} {name} {args}')
        if func.__doc__:
            print(func.__doc__)
    
    usage = f'''USAGE:
    {sys.argv[0]} <command> <arguments>
    Commands: [ {' | '.join(functions.keys())} ]
    {sys.argv[0]} -h <command>'''
    
    if len(sys.argv) == 1:
        print(usage)
        return 1
    
    command = sys.argv[1]
    
    # Handle help
    if command == '-h':
        if len(sys.argv) == 2:
            print(usage)
        else:
            func_name = sys.argv[2]
            if func_name not in functions:
                print(f'Unknown command: {func_name}')
                return 1
            print_help(functions[func_name], func_name)
        return 0
    
    # Handle commands
    if command not in functions:
        print(f'Unknown command: {command}')
        print(usage)
        return 1
    
    func = functions[command]
    args = sys.argv[2:]
    
    # Check argument count
    spec = inspect.getfullargspec(func)
    n_required = len(spec.args) - len(spec.defaults or [])
    n_max = len(spec.args)
    
    if not (n_required <= len(args) <= n_max):
        print_help(func, command)
        return 1
    
    # Call function
    try:
        func(*args)
        return 0
    except Exception as e:
        Log.error(str(e))
        return 1


if __name__ == '__main__':
    sys.exit(_cli_main())
