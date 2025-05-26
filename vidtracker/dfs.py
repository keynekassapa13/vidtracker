"""
Re-implementation of paper 
Distribution Fields for Tracking
Sevilla-Lara et al. (2012)
"""

import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
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
    pad_h = kh - 1
    pad_w = kw - 1
    img_p = np.pad(img, 
                   ((pad_h, pad_h),
                    (pad_w, pad_w)),
                   mode='constant', constant_values=0).astype(np.float32)
    ker_p = np.zeros_like(img_p, dtype=np.float32)
    ker_p[:kh, :kw] = kernel
    ker_p = np.roll(np.roll(ker_p, -kh//2, axis=0), -kw//2, axis=1)
    # Convolve in frequency domain
    # Transform back to spatial domain
    # and take the real part
    fft_img = np.fft.fft2(img_p)
    fft_ker = np.fft.fft2(ker_p)
    conv_p = np.fft.ifft2(fft_img * fft_ker).real
    start_h = pad_h - (kh // 2)
    start_w = pad_w - (kw // 2)
    out = conv_p[start_h:start_h+h, start_w:start_w+w]
    return out

class DFSTracker:
    def __init__(self, first_frame, init_bbox, cfg):
        """
        Args:
            first_frame: The first frame of the video.
            init_bbox: The initial bounding box of the object to track.
            cfg: Configuration object containing parameters
                cfg.DFS.bins: Number of bins for the histogram
                cfg.DFS.sigma_s_list: List of spatial smoothing parameters
                cfg.DFS.sigma_f: Feature smoothing parameter
                cfg.DFS.lambda_: Update rate for the model
        """
        x, y, w, h = init_bbox
        self.num_workers   = cfg.NUM_WORKERS
        self.sample        = cfg.SAMPLE

        self.bins          = cfg.DFS.bins
        self.sigma_s_list  = cfg.DFS.sigma_s_list
        self.sigma_f       = cfg.DFS.sigma_f
        self.lambda_       = cfg.DFS.lambda_
        self.w, self.h     = w, h
        self.cx = x + w//2
        self.cy = y + h//2

        self.angle = 0.0
        self.s_rad = cfg.DFS.s_rad
        self.max_delta = cfg.DFS.max_delta
        self.num_angles = cfg.DFS.num_angles
        self.dynamic = cfg.DFS.dynamic
        self.dyn_scale = cfg.DFS.dyn_scale
        self.dyn_tries = cfg.DFS.dyn_tries

        gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
        patch = gray[y:y+h, x:x+w]
        self.models = self.build_model(patch)
    
    
    def smooth_spatial(self, df, sigma_s):
        """
        Smooth the distribution field in the spatial domain using a Gaussian kernel.
        Axis 0 and 1 are the spatial dimensions
        Args:
            df: distribution field representation of the patch
            sigma_s: spatial smoothing parameter
        Returns:
            out: smoothed distribution field
        """
        ker = gaussian_kernel(sigma_s)
        h, w, b = df.shape
        out = np.zeros_like(df)
        for k in range(b):
            out[:, :, k] = convolve(df[:, :, k], ker)
        return out
    
    def smooth_feature(self, df):
        """
        Smooth the distribution field across intensity bins
        Reduces sensitivity to small intensity changes
        Axis 2 is the intensity bin.
        Args:
            df: distribution field representation of the patch
        Returns:
            out: smoothed distribution field
        """
        sigma = self.sigma_f
        r = int(3 * sigma)
        ax = np.arange(-r, r+1, dtype=np.float32)
        # 1D gaussian kernel
        kernel = np.exp(-ax**2 / (2 * sigma**2))
        kernel /= np.sum(kernel)
        # smooth histogram bins
        h, w, b = df.shape
        pad = np.pad(df, ((0, 0), (0, 0), (r, r)), mode='constant')
        out = np.zeros_like(df)
        for i in range(b):
            # pad[:, :, i:i+2*r+1] is the window of size (2*r+1)
            window = pad[:, :, i:i+2*r+1]
            # window size (h, w, 2*r+1)
            # kernel size (2*r+1,)
            # out[:, :, i] is the smoothed bin
            out[:, :, i] = np.tensordot(window, kernel, axes=([2], [0]))
        return out
    
    def build_model(self, patch):
        """
        Args:
            patch: The patch of the object to track.
        Returns:
            models: distribution field representation of an image patch
        """
        base = explode(patch, self.bins)
        models = []
        for sigma_s in self.sigma_s_list:
            sm = self.smooth_spatial(base, sigma_s)
            sm = self.smooth_feature(sm)
            models.append(sm)
        return models
    
    
    def process_frame(self, frame):
        """
        Args:
            frame: current frame of the video
        Returns:
            loc: updated location of the bounding box
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        H, W = gray.shape

        angles = np.linspace(self.angle - self.max_delta,
                            self.angle + self.max_delta,
                            self.num_angles)

        scales = (
                    np.linspace(1.0 - self.dyn_scale,
                                1.0 + self.dyn_scale,
                                self.dyn_tries)
                    .tolist()
                    if self.dynamic
                    else [1.0]
                )
        # 1. Build all candidate tasks
        tasks = []
        for s in scales:
            w_s, h_s = int(self.w*s), int(self.h*s)
            half_w, half_h = w_s//2, h_s//2
            for angle in angles:
                # warp once per (s, theta)
                M = cv2.getRotationMatrix2D((self.cx, self.cy), angle, 1.0)
                warped = cv2.warpAffine(gray, M, (W, H),
                                        flags=cv2.INTER_LINEAR,
                                        borderMode=cv2.BORDER_REPLICATE)
                for dx in range(-self.s_rad, self.s_rad+1, 4):
                    for dy in range(-self.s_rad, self.s_rad+1, 4):
                        x0 = int(self.cx - half_w) + dx
                        y0 = int(self.cy - half_h) + dy
                        if x0 < 0 or y0 < 0 or x0 + w_s > W or y0 + h_s > H:
                            continue
                        tasks.append((warped, x0, y0, w_s, h_s, s, angle))

        n = min(self.sample, len(tasks))
        tasks = random.sample(tasks, n)

        # 2. Define worker function
        def eval_patch(args):
            warped, x0, y0, w_s, h_s, s, theta = args
            patch = warped[y0:y0+h_s, x0:x0+w_s]
            patch = cv2.resize(patch, (self.w, self.h))
            df_base = explode(patch, self.bins)
            cost = 0.0
            for sigma_s, model in zip(self.sigma_s_list, self.models):
                df_s = self.smooth_spatial(df_base, sigma_s)
                df_s = self.smooth_feature(df_s)
                cost += np.sum(np.abs(df_s - model))

            return cost, (x0, y0, s, theta)

        best_cost = np.inf
        best_params = None
        # for task in tasks:
        #     cost, params = eval_patch(task)
        #     if cost < best_cost:
        #         best_cost, best_params = cost, params
        
        # 3. Run in parallel
        with ThreadPoolExecutor(max_workers=self.num_workers) as pool:
            futures = [pool.submit(eval_patch, t) for t in tasks]
            for f in as_completed(futures):
                cost, params = f.result()
                if cost < best_cost:
                    best_cost, best_params = cost, params

        # 4. Pick the best
        if best_params is not None:
            x0, y0, s, θ = best_params
            self.angle = θ
            self.cx = x0 + int(self.w*s/2)
            self.cy = y0 + int(self.h*s/2)
            gray_patch = gray[y0:y0+int(self.h*s), x0:x0+int(self.w*s)]
            gray_patch = cv2.resize(gray_patch, (self.w, self.h))
            new_models = self.build_model(gray_patch)
            # 5. Update status 
            for i in range(len(self.models)):
                self.models[i] = ( self.lambda_*self.models[i]
                                  + (1-self.lambda_)*new_models[i] )

        return self.cx, self.cy, int(self.w*s), int(self.h*s), self.angle
