import cv2
import numpy as np
import random
import math
from loguru import logger

class HaarFeature:
    def __init__(self, win_w, win_h, min_rects=2, max_rects=6, use_channels=None):
        """
        Args:
            win_w, win_h: window width and height
            min_rects, max_rects: minimum and maximum number of rectangles
            use_channels: randomly select a channel from this list, 
                           or None to use all channels
        """
        self.win_w, self.win_h = win_w, win_h
        self.min_rects, self.max_rects = min_rects, max_rects
        if use_channels:
            self.channel = random.choice(use_channels)
        else:
            self.channel = None
        n = random.randint(self.min_rects, self.max_rects)
        self.rects = []
        self.weights = []
        self._max_sum = 0.0
        for _ in range(n):
            w = random.random() * 2 - 1
            x1 = random.randint(0, win_w - 3)
            y1 = random.randint(0, win_h - 3)
            w_rect = random.randint(1, win_w - x1 - 2)
            h_rect = random.randint(1, win_h - y1 - 2)
            self.rects.append((x1, y1, w_rect, h_rect))
            self.weights.append(w)
            self._max_sum += abs(w * (w_rect + 1) * (h_rect + 1) * 255.0)

    def compute(self, ii_list, x0, y0):
        """
        Args:
            ii_list: integral image list, one for each channel
            x0, y0: top-left corner of the window
        Returns:
            val: computed feature value
        """
        if self.channel is not None:
            ii = ii_list[self.channel]
        else:
            ii = ii_list
        val = 0.0
        for (x1, y1, w_rect, h_rect), w in zip(self.rects, self.weights):
            xa, ya = x0 + x1, y0 + y1
            xb, yb = xa + w_rect + 1, ya + h_rect + 1
            s = ii[yb, xb] - ii[ya, xb] - ii[yb, xa] + ii[ya, xa]
            val += w * s
        return val / (self._max_sum + 1e-12)

class WeakClassifier:
    def __init__(self, feature, lr=0.85):
        """
        Args:
            feature: HaarFeature object
            lr: learning rate for updating mean and std
        """
        self.feature = feature
        self.lr = lr
        self.mu_pos = 0.0
        self.sigma_pos = 1.0
        self.mu_neg = 0.0
        self.sigma_neg = 1.0

    def update(self, feats, labels):
        """
        Args:
            feats: feature values
            labels: corresponding labels (1 for positive, 0 for negative)
        """
        pos = feats[labels == 1]
        neg = feats[labels == 0]
        if len(pos) > 0:
            mu1 = np.mean(pos)
            s1 = np.std(pos) + 1e-6
            self.mu_pos    = self.lr * self.mu_pos    + (1 - self.lr) * mu1
            self.sigma_pos = self.lr * self.sigma_pos + (1 - self.lr) * s1
        if len(neg) > 0:
            mu0 = np.mean(neg)
            s0 = np.std(neg) + 1e-6
            self.mu_neg    = self.lr * self.mu_neg    + (1 - self.lr) * mu0
            self.sigma_neg = self.lr * self.sigma_neg + (1 - self.lr) * s0

    def score(self, val):
        """
        Args:
            val: feature value
        Returns:
            log-odds ratio of the feature value
        """
        # p1 = probability(val | positive)
        p1 = math.exp(-0.5 * ((val - self.mu_pos) / self.sigma_pos) ** 2) / (self.sigma_pos * math.sqrt(2 * math.pi))
        # p0 = probability(val | negative)
        p0 = math.exp(-0.5 * ((val - self.mu_neg) / self.sigma_neg) ** 2) / (self.sigma_neg * math.sqrt(2 * math.pi))
        # log odds (logit)
        return math.log((p1 + 1e-12) / (p0 + 1e-12))
