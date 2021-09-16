import dtcwt
import numpy as np
from numpy.lib.stride_tricks import as_strided
import pyyawt
import pywt


def _NeighCoeffScaling(real_coeff, sigma, lam):
    """
    Get scaling factor using NeighCoeff
    """
    L = 3
    s = real_coeff.strides[0]
    padded = np.pad(real_coeff, (1, 1), mode='edge')
    blocks = as_strided(padded, shape=(len(padded) - 2, 3), strides=(s, s))
    fun1d = lambda y: 1 - (sigma**2 / np.sum(y**2)) * lam * L
    scaling = np.apply_along_axis(fun1d, 1, blocks)
    scaling[scaling < 0] = 0
    return scaling


def _MAD(x):
    """
    Median absolute deviation
    """
    md = np.median(x)
    mad = np.median(np.abs(x - md))
    return mad


def DTCWTNeighCoeff(x, nlevels=0, sigma_est='all', noise_level=0):
    """
    Dual Tree Complex Wavelet Transform Neighbouring Coefficients Filter
    Uses DTCWT to compute complex coefficients
    The stdev of coefficients is estimated based on the given noies_level and sigma_est parameters
    Coefficients are locally considered noise if they are surrounded by mostly noise.
    Noise coefficients in both real and imag parts are removed and the resulting signal reconstructed.

    Parameters
    ----------
    x: np.ndarray
        1D signal array to be filtered by DTCWT. Must be even length, i.e. len(x)%2 = 0
    nlevels: int, optional
        Number of levels to perform DTCWT. Must be less than or equal to log2(len(x)).
        Defaults to int(log2(len(x))) if 0 is passed.
    sigma_est: string ['all', 'non-zero'] or float, optional
        If a float is passed, this value will be used as the sigma estimate for both real and imaginary coefficients
        Estimate of noise standard deviation, sigma.
        If string is passed, sigma is estimated based on the transform coefficients at noise_level.
        On very sparse signals with little noise, 'non-zero' should be used. Otherwise, 'all' should be used.
    noise_level: int, optional
        Which decomposition band to estimate the noise standard deviation from, if no numeric value is passed in
        in sigma_est. Defaults to 0, which is the first (highest frequency) level.

    Returns
    -------
    recon: np.ndarray
        Denoised signal after reconstructing filtered DTCWT coefficients
    Raises
    ------
    ValueError:
        nlevels is greater than log2(len(x)) or less than 0
        sigma_est is pass an invalid value that is not a float or specified string
        x is not of even length
    """
    # Input checking for nlevels
    if nlevels > np.log2(len(x)):
        raise ValueError(f"Number of decomposition levels ({nlevels}) must be at most {int(np.log2(len(x)))}"
                         f"for signal of length {len(x)}.")
    elif nlevels < 0:
        raise ValueError(f"Number of decomposition levels ({nlevels}) must be positive.")
    if nlevels == 0:
        nlevels = int(np.log2(len(x)))  # int default floors

    # Do transform
    transform = dtcwt.Transform1d()
    pyramid = transform.forward(x, nlevels=nlevels)
    hpc = pyramid.highpasses

    # Noise std estimate given numerical value
    if isinstance(sigma_est, float):
        sigma_re = sigma_est
        sigma_im = sigma_est
    # Estimate noise std if string given
    elif isinstance(sigma_est, str):
        # Input checking for noise_level
        if not (0 <= noise_level < nlevels):
            raise ValueError(f"noise_level must be between 0 and {nlevels - 1}")
        if sigma_est == 'all':
            hpc_re = np.abs(hpc[noise_level].real)
            hpc_im = np.abs(hpc[noise_level].imag)
        elif sigma_est == 'non-zero':
            hpc_re = np.abs(hpc[noise_level].real)
            hpc_re = hpc_re[hpc_re != 0]
            hpc_im = np.abs(hpc[noise_level].imag)
            hpc_im = hpc_im[hpc_im != 0]
        else:
            raise ValueError(f"sigma_est was {sigma_est} but must be a float or string \'all\' or \'non-zero\'")
        sigma_re = (1/0.6745) * _MAD(hpc_re)
        sigma_im = (1/0.6745) * _MAD(hpc_im)
    else:
        raise ValueError(f"sigma_est was {sigma_est} but must be a float or string \'all\' or \'non-zero\'")

    # Vars for shrinkage
    lam = 2/3 * np.log(len(x))
    hpc_filt = []

    # Scale complex coefficients
    for cc in hpc:
        re = cc.real.flatten()
        im = cc.imag.flatten()

        re_filt = _NeighCoeffScaling(re, sigma_re, lam) * re
        im_filt = _NeighCoeffScaling(im, sigma_im, lam) * im
        cc_filt = re_filt + 1j * im_filt
        cc_filt = cc_filt[:, np.newaxis]
        hpc_filt.append(cc_filt)

    # Reconstruct signal
    pyramid_recon = dtcwt.Pyramid(pyramid.lowpass, hpc_filt)
    recon = transform.inverse(pyramid_recon)
    return recon


def direct_tvd(x, weight=0.1):
    """
    Implements total variation denoising using a non-iterative algorithm described in Condat, 2013 doi:10.1109/LSP.2013.2278339. Pure Python implementation.

    The algorithm seeks to construct segments of constant values and detects changepoints when the signal varies too much from these segements to construct the next segment. This algorithm essentially tries to construct a piecewise step function whenever possible.
    Parameters
    ----------
        x: np.ndarray
            1D signal input array to denoise
        weight: float
            Weight (lambda) parameter in the total variation algorithm that controls smoothness of the denoising

    Returns
    -------
        y: np.ndarray
            1D signal array that has been denoised
    """
    N_ret = x.shape[0]

    # Pad if size is big enough
    if x.size > 10:    
        x = np.pad(x, (0, 10), mode='reflect')
    
    N = x.shape[0]
    lam = weight
    v_min = x[0] - lam
    v_max = x[0] + lam
    u_min = lam
    u_max = -lam
    k = 0
    k_0 = 0
    k_p = 0
    k_m = 0  
    y = np.zeros_like(x)

    # 2 and 7: Termination condition
    while k < N - 1:
        # 3: Next point cannot be added to segment, v_min too high
        if x[k + 1] + u_min < v_min - lam:
            y[k_0:k_m + 1] = v_min
            k = k_0 = k_m = k_p = k_m + 1
            v_min = x[k]
            v_max = x[k] + 2 * lam
            u_min = lam
            u_max = -lam
        # 4: Next point cannot be added to segment, v_max too low
        elif x[k + 1] + u_max > v_max + lam:
            y[k_0:k_p + 1] = v_max
            k = k_0 = k_m = k_p = k_p + 1
            v_min = x[k] - 2 * lam
            v_max = x[k]
            u_min = lam
            u_max = -lam
        # 5: No jump necessary, add point to segment
        else:
            k = k + 1
            u_min = u_min + x[k] - v_min
            u_max = u_max + x[k] - v_max
            # 6a: Update bounds v_min
            if u_min >= lam:
                v_min = v_min + (u_min - lam)/(k - k_0 + 1)
                u_min = lam
                k_m = k
            # 6b: Update bounds v_max
            if u_max <= -lam:
                v_max = v_max + (u_max + lam)/(k - k_0 + 1)
                u_max = -lam
                k_p = k
        if k == N - 1:
            # 8: v_min too high and we do a negative step
            if u_min < 0:
                y[k_0:k_m + 1] = v_min
                k = k_0 = k_m = k_m + 1
                v_min = x[k]
                u_min = lam
                u_max = x[k] + lam - v_max
            # 9: v_max too low, do a positive step
            elif u_max > 0:
                y[k_0:k_p + 1] = v_max
                k = k_0 = k_p = k_p + 1
                v_max = x[k]
                u_max = -lam
                u_min = x[k] - lam - v_min
            # 10: Construct segment to end of signal
            else:
                y[k_0:N] = v_min + u_min/(k - k_0 + 1)
                return y[:N_ret]
    # Exited while loop, so k = N-1
    # 2 after loop
    y[k] = v_min + u_min
    return y[:N_ret]

def wden_softSURE(x, wavelet='db4', level=1):
    """
    Parameters:
    -----------
    x: np.ndarray
    wavelet: str
        String describing which wavelet to use in the transform
    level: int
        Which level to use for estimating noise
    """
    [XD,CXD,LXD] = pyyawt.denoising.wden(x, 'rigrsure', 's', 'mln', level, wavelet)
    return XD

def madev(d, axis=None):
    """ Mean absolute deviation of a signal """
    return np.mean(np.absolute(d - np.mean(d, axis)), axis)


def wavelet_denoising(x, wavelet='db4', level=1):
    coeff = pywt.wavedec(x, wavelet, mode="per")
    sigma = (1/0.6745) * madev(coeff[-level])
    uthresh = sigma * np.sqrt(2 * np.log(len(x)))
    coeff[1:] = (pywt.threshold(i, value=uthresh, mode='hard') for i in coeff[1:])
    return pywt.waverec(coeff, wavelet, mode='per')
