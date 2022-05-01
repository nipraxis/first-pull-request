import numpy as np
import nibabel as nib
import nipraxis

from numpy.typing import ArrayLike


def spm_global(vol: ArrayLike) -> ArrayLike:
    """Calculate SPM global metric for array `vol`.

    Parameters
    ----------
    vol : ArrayLike
        Array giving image data, usually 3D.

    Returns
    -------
    ArrayLike
        SPM global metric for `vol`
    """
    T = np.mean(vol) / 8
    
    return np.mean(vol[vol > T])


def get_spm_globals(file_name: str) -> ArrayLike:
    """Calculate SPM global metrics for volumes in image filename `file_name`.

    Parameters
    ----------
    file_name : str
        Filename of file containing 4D image

    Returns
    -------
    spm_vals : ArrayLike
        SPM global metric for each 3D volume in the 4D image.
    """
    img = nib.load(file_name)
    data = img.get_fdata()
    spm_vals = np.empty(img.shape[-1])
    for i, vol in enumerate(range(img.shape[-1])):
        spm_vals[i] = spm_global(data[..., vol])

    return spm_vals


def main():
    bold_fname = nipraxis.fetch_file('ds107_sub012_t1r2.nii')
    glob_vals = get_spm_globals(bold_fname)
    if glob_vals is None:
        raise ValueError('Did you return your global values?')
    expected_values = np.loadtxt('global_signals.txt')
    if np.allclose(glob_vals, expected_values, rtol=1e-4):
        print('OK: your values and SPMs are close')
    else:
        print('SPM and your values differ')
        print('Yours:', [float(v) for v in glob_vals])
        print('SPMs:', expected_values)


if __name__ == '__main__':
    main()
