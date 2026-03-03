"""
boiler plate code, partially generated with ChatGPT
"""

import numpy as np
import numpy.typing as npt
import os
import re
from typing import List

from transfer_function import TransferFunction

def load_neff(path : str) ->  tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    """
    Loads frequency and n_eff from a file exported from Lumerical FDE (Frequency Sweep)
    @path to the file
    @return f, Re(n_eff(f))
    """
    with open(path, 'r') as f:
        lines = f.readlines()

    # Find start of n_eff section
    start_idx = None
    for i, line in enumerate(lines):
        if line.strip().startswith("f,neff(real),neff(imag)"):
            start_idx = i + 1
            break

    if start_idx is None:
        raise ValueError("Could not find n_eff section in file.")

    # Find end of n_eff section
    end_idx = len(lines)
    for j in range(start_idx, len(lines)):
        if lines[j].strip().startswith("f_vg") or lines[j].strip().startswith("f_D"):
            end_idx = j
            break

    # Load only that section into numpy
    data = np.loadtxt(lines[start_idx:end_idx], delimiter=',')

    f = data[:, 0]
    neff_real = data[:, 1]

    return f, neff_real

def generate_and_save_transfer_functions(
    save_path: str,
    N_puf: int,
    start_wavelength: float,
    stop_wavelength: float,
    f0: float,
    neff0: float,
    f1: float,
    neff1: float,
    f2: float,
    neff2: float,
    grating_length: float,
    manufacturing_variations_segment_length: float,
    force: bool = False
) -> List[TransferFunction]:
    """
    Generate, save, and return multiple moiré grating TransferFunction objects.
    Aborts safely if any target file already exists (unless force=True).

    Parameters
    ----------
    save_path : str
        Directory to save the .npz files (will be created if it doesn't exist).
    N_puf : int
        Number of transfer functions to generate.
    start_wavelength, stop_wavelength : float
        Wavelength range.
    f0, f1, f2 : float
        Grating spatial frequencies.
    neff0, neff1, neff2 : float
        Effective indices corresponding to the frequencies.
    grating_length : float
        Total length of the grating.
    manufacturing_variations_segment_length : float
        Variation in segment length (for moiré effect).
    force : bool, optional
        If True, existing files will be overwritten. Default is False.

    Returns
    -------
    List[TransferFunction]
        A list of the generated TransferFunction instances.

    Notes
    -----
    - If force=False and *any* target file exists, generation is aborted
      and an empty list is returned.
    """
    os.makedirs(save_path, exist_ok=True)

    # Check for existing files if not forcing overwrite
    if not force:
        existing_files = [
            os.path.join(save_path, f"moire_grating_{p}.npz")
            for p in range(N_puf)
            if os.path.exists(os.path.join(save_path, f"moire_grating_{p}.npz"))
        ]
        if existing_files:
            print("❌ Aborted: The following files already exist:")
            for f in existing_files:
                print(f"   {f}")
            print("Use force=True to overwrite.")
            return []

    if force:
        print(f"⚠️ Overwrite mode enabled: existing files will be replaced.")

    print(f"✅ Starting generation of {N_puf} transfer functions...")

    transfer_functions = []
    for p in range(N_puf):
        tf = TransferFunction.moireGrating(
            start_wavelength=start_wavelength,
            stop_wavelength=stop_wavelength,
            f0=f0, neff0=neff0,
            f1=f1, neff1=neff1,
            f2=f2, neff2=neff2,
            length=grating_length,
            delta_L=manufacturing_variations_segment_length
        )

        filename = os.path.join(save_path, f"moire_grating_{p}.npz")
        tf.save_npz(filename)
        transfer_functions.append(tf)
        print(f"💾 Saved: {filename}")

    print(f"✅ Done. {N_puf} transfer functions saved in '{save_path}'")
    return transfer_functions


def load_all_transfer_functions(save_path: str, pattern: str = r"moire_grating_(\d+)\.npz") -> List[TransferFunction]:
    """
    Automatically load all TransferFunction objects from .npz files in a folder.

    Parameters
    ----------
    save_path : str
        Directory containing saved .npz files.
    pattern : str, optional
        Regex pattern to match filenames (default: 'moire_grating_(\\d+)\\.npz').

    Returns
    -------
    List[TransferFunction]
        List of loaded TransferFunction instances sorted by index.
    """
    files = []
    regex = re.compile(pattern)

    # Find all matching files and extract their indices
    for fname in os.listdir(save_path):
        match = regex.match(fname)
        if match:
            idx = int(match.group(1))
            files.append((idx, os.path.join(save_path, fname)))

    # Sort by numeric index
    files.sort(key=lambda x: x[0])

    # Load all transfer functions
    transfer_functions = []
    for idx, filepath in files:
        try:
            tf = TransferFunction.load_npz(filepath)
            transfer_functions.append(tf)
        except Exception as e:
            print(f"⚠️ Could not load {filepath}: {e}")

    print(f"✅ Loaded {len(transfer_functions)} transfer functions from '{save_path}'")
    return transfer_functions