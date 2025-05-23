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

"""
MILBoost - Multiple Instance Learning with Boosting
Boosting algorithm goal is to combine many weak classifier into additive strong classifier
"""
class OnlineMILBoost:
    def __init__(self, win_w, win_h, M=250, K=50, lr=0.85):
        """
        Args:
            win_w, win_h: window width and height
            M: number of weak classifiers
            K: number of selected classifiers
            lr: learning rate for updating mean and std
        """
        self.win_w, self.win_h = win_w, win_h
        self.pool = [WeakClassifier(HaarFeature(win_w, win_h), lr) for _ in range(M)]
        self.M, self.K = M, K
        self.selected = []

    @staticmethod
    def sigmoid(x):
        if x >= 0:
            z = math.exp(-x)
            return 1.0 / (1.0 + z)
        else:
            z = math.exp(x)
            return z / (1.0 + z)

    def train_frame(self, ii, pos_centers, neg_centers):
        """
        Args:
            ii: integral image
            pos_centers: list of positive center coordinates
            neg_centers: list of negative center coordinates
        Update:
            self.pool: list of weak classifiers
            self.selected: list of selected classifiers
        """
        H_img, W_img = ii.shape[0]-1, ii.shape[1]-1
        hw, hh = self.win_w // 2, self.win_h // 2
        coords, labels = [], []

        # Extract pos patches
        for cx, cy in pos_centers:
            x0, y0 = int(cx - hw), int(cy - hh)
            if 0 <= x0 <= W_img - self.win_w and 0 <= y0 <= H_img - self.win_h:
                coords.append((x0, y0))
                labels.append(1)
        # Extract neg patches
        for cx, cy in neg_centers:
            x0, y0 = int(cx - hw), int(cy - hh)
            if 0 <= x0 <= W_img - self.win_w and 0 <= y0 <= H_img - self.win_h:
                coords.append((x0, y0))
                labels.append(0)
        
        if not coords:
            return
        
        N_pos = sum(labels)
        N = len(coords)
        L = np.array(labels, dtype=np.int32)
        # Feature matrix
        all_feats = np.zeros((self.M, N))

        for m, h in enumerate(self.pool):
            # m is the index of the weak classifier
            # h is the weak classifier
            # each weak classifier compute its feature value for every patch
            for i, (x0, y0) in enumerate(coords):
                all_feats[m, i] = h.feature.compute(ii, x0, y0)
            # each weak classifier updates the mean and std
            h.update(all_feats[m], L)

        # Select K weak classifiers
        # Algo 2 in paper
        H_sel = np.zeros(N)         # H_sel is the sum of the selected weak classifiers
        self.selected = []          # selected weak classifiers
        for _ in range(self.K):
            best_h, best_ll = None, -np.inf
            for idx, h in enumerate(self.pool):
                if h in self.selected:
                    continue

                ll = 0.0
                one_minus = 1.0
                for i in np.where(labels == 1)[0]:
                    s = self.sigmoid(H_sel[i] + h.score(all_feats[idx, i]))
                    one_minus *= (1.0 - s)
                ll += math.log(1.0 - one_minus + 1e-12)

                for i in np.where(labels == 0)[0]:
                    s = self.sigmoid(H_sel[i] + h.score(all_feats[idx, i]))
                    ll += math.log(1.0 - s + 1e-12)

                # argmax
                if ll > best_ll:
                    best_ll, best_h = ll, h
            if best_h is None:
                break
            self.selected.append(best_h)        # add the best weak classifier to the selected list
            bidx = self.pool.index(best_h)      # get the index of the best weak classifier
            # Update the selected weak classifier
            H_sel += np.array([best_h.score(v) for v in all_feats[bidx]])

    def score_patch(self, ii, cx, cy):
        """
        Probability of the patch being positive
        Args:
            ii: integral image
            cx, cy: center coordinates of the patch
        Returns:
            Hval: computed score for the patch
        """
        hw, hh = self.win_w // 2, self.win_h // 2
        x0, y0 = int(cx - hw), int(cy - hh)
        Hval = 0.0
        for h in self.selected:
            # h is the weak classifier
            # h.feature is the Haar feature
            # h,feature.compute is the feature value for the patch
            # Hval is the sum of the feature values - in log-odds space
            Hval += h.score(h.feature.compute(ii, x0, y0))
        # Instance Probability
        return self.sigmoid(Hval)