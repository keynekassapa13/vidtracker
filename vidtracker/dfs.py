"""
Re-implementation of paper 
Distribution Fields for Tracking
Sevilla-Lara et al. (2012)
"""

import cv2
import numpy as np
import random

# Explode function
def explode(patch, bins):
    """
    Explodes the patch (2D) into a distribution field representation (3D).
    This is done by creating a histogram of the pixel values in the patch.
    The histogram is then converted into a distribution field.
    Args:
        patch: The patch of the object to track.
        bins: Number of bins for the histogram
    Returns:
        df: distribution field representation of the patch
    """
    h, w = patch.shape
    df = np.zeros((h, w, bins), dtype=np.float32)
    # inds gives the bin index for pixel at (y, x)
    inds = np.clip((patch.astype(np.float32) / 255.0 * (bins-1)).astype(np.int32), 0, bins - 1)
    # one hot encoding
    # df[y, x, k] = 1 if pixel at (y, x) belongs to bin k
    # binary mask for each bin
    for k in range(bins):
        df[:, :, k] = (inds == k).astype(np.float32)
    return df


def gaussian_kernel(sigma):
    """
    Gaussian kernel for spatial smoothing.
    Args:
        sigma: Standard deviation for the Gaussian kernel
    Returns:
        k: Gaussian kernel, sized (2*r+1, 2*r+1)
    """
    r = int(3 * sigma)
    ax = np.arange(-r, r+1, dtype=np.float32)
    xx, yy = np.meshgrid(ax, ax)
    k = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    k /= np.sum(k)
    return k

def convolve(img, kernel):
    """
    Convolve the image with the kernel using FFT.
    Args:
        img: The input image
        kernel: The kernel to convolve with
    Returns:
        out: Convolved image
    """
    h, w = img.shape
    kh, kw = kernel.shape
    pad = np.zeros((h, w), dtype=np.float32)
    pad[:kh, :kw] = kernel
    pad = np.roll(np.roll(pad, -kh//2, axis=0), -kw//2, axis=1)
    
    # Perform FFT convolution
    # Convolve in frequency domain
    # Transform back to spatial domain
    # and take the real part
    return np.fft.ifft2(np.fft.fft2(img) * np.fft.fft2(pad)).real
