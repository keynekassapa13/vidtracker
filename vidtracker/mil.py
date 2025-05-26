"""
Re-implementation of paper 
Visual Tracking with Online Multiple Instance Learning 
Babenko et al. (2009)
"""

import cv2
import numpy as np
import random
import math
class HaarFeature:
    def __init__(self, win_w, win_h, min_rects=2, max_rects=30, use_channels=None):
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
        self.rel_rects = []  # relative rectangles: (x_ratio, y_ratio, w_ratio, h_ratio)
        self.weights = []
        self._max_sum = 0.0

        for _ in range(n):
            w = random.random() * 2 - 1
            x1 = random.uniform(0, 0.9)
            y1 = random.uniform(0, 0.9)
            w_rect = random.uniform(0.05, 1.0 - x1)
            h_rect = random.uniform(0.05, 1.0 - y1)
            self.rel_rects.append((x1, y1, w_rect, h_rect))
            self.weights.append(w)
            self._max_sum += abs(w * (w_rect * win_w + 1) * (h_rect * win_h + 1) * 255.0)


    def compute(self, ii_list, x0, y0, win_w=None, win_h=None):
        """
        Args:
            ii_list: integral image list, one for each channel
            x0, y0: top-left corner of the window
            win_w, win_h: window width and height (optional)
        Returns:
            val: computed feature value
        """
        if win_w is None: win_w = self.win_w
        if win_h is None: win_h = self.win_h

        if self.channel is not None:
            ii = ii_list[self.channel]
        else:
            ii = ii_list

        val = 0.0
        for (rx, ry, rw, rh), w in zip(self.rel_rects, self.weights):
            x1 = int(rx * win_w)
            y1 = int(ry * win_h)
            w_rect = int(rw * win_w)
            h_rect = int(rh * win_h)
            xa, ya = x0 + x1, y0 + y1
            xb = min(ii.shape[1] - 1, xa + w_rect + 1)
            yb = min(ii.shape[0] - 1, ya + h_rect + 1)
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
    def __init__(self, win_w, win_h, cfg):
        """
        Args:
            win_w, win_h: window width and height
            cfg: configuration object with parameters
                cfg.MIL.M: number of weak classifiers
                cfg.MIL.K: number of selected classifiers
                cfg.MIL.lr: learning rate for updating mean and std
        """
        self.win_w, self.win_h = win_w, win_h
        self.pool = [WeakClassifier(HaarFeature(win_w, win_h, min_rects=cfg.MIL.min_rects, max_rects=cfg.MIL.max_rects), cfg.MIL.lr) for _ in range(cfg.MIL.M)]
        self.M, self.K = cfg.MIL.M, cfg.MIL.K
        self.selected = []

    @staticmethod
    def sigmoid(x):
        if x >= 0:
            z = math.exp(-x)
            return 1.0 / (1.0 + z)
        else:
            z = math.exp(x)
            return z / (1.0 + z)

    def train_frame(self, ii, pos_centers, neg_centers, win_w=None, win_h=None):
        """
        Args:
            ii: integral image
            pos_centers: list of positive center coordinates
            neg_centers: list of negative center coordinates
        Update:
            self.pool: list of weak classifiers
            self.selected: list of selected classifiers
        """
        # Update window size
        if win_w is None: win_w = self.win_w
        if win_h is None: win_h = self.win_h
        self.win_w, self.win_h = win_w, win_h

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
                all_feats[m, i] = h.feature.compute(ii, x0, y0, win_w=win_w, win_h=win_h)
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

    def score_patch(self, ii, cx, cy, win_w=None, win_h=None):
        """
        Probability of the patch being positive
        Args:
            ii: integral image
            cx, cy: center coordinates of the patch
        Returns:
            Hval: computed score for the patch
        """
        if win_w is None: win_w = self.win_w
        if win_h is None: win_h = self.win_h

        hw, hh = win_w // 2, win_h // 2
        x0, y0 = int(cx - hw), int(cy - hh)
        Hval = 0.0

        for h in self.selected:
            # h is the weak classifier
            # h.feature is the Haar feature
            # h,feature.compute is the feature value for the patch
            # Hval is the sum of the feature values - in log-odds space
            Hval += h.score(h.feature.compute(ii, x0, y0, win_w=win_w, win_h=win_h))
        
        # Instance Probability
        return self.sigmoid(Hval)
    

class MILTracker:
    def __init__(self, first_frame, init_bbox, cfg):
        """
        Args:
            first_frame: first frame of the video
            init_bbox: initial bounding box (x, y, width, height)
            cfg: configuration object with parameters
                MIL.s_rad: search radius (candidate locations)
                MIL.pos_rad: positive sample radius
                MIL.neg_rad: negative sample radius
                MIL.num_neg: number of negative samples
                MIL.lr: learning rate for updating mean and std
                MIL.dynamic: whether to use dynamic scale
                MIL.dyn_scale: scale factor for dynamic scaling
                MIL.dyn_tries: number of tries for dynamic scale
                MIL.num_angles: number of angles to sample
                MIL.max_delta: maximum angle delta for rotation
        """
        x, y, w, h = init_bbox
        self.w, self.h = w, h
        self.loc = (x + w // 2, y + h // 2)         # center of the bounding box
        self.angle = 0.0 # angle

        self.num_workers   = cfg.NUM_WORKERS
        self.sample        = cfg.SAMPLE
        
        self.s_rad = cfg.MIL.s_rad
        self.pos_rad = cfg.MIL.pos_rad
        self.neg_rad = cfg.MIL.neg_rad
        self.num_neg = cfg.MIL.num_neg

        self.dynamic = cfg.MIL.dynamic
        self.dyn_scale = cfg.MIL.dyn_scale
        self.dyn_tries = cfg.MIL.dyn_tries
        self.num_angles = cfg.MIL.num_angles
        self.max_delta = cfg.MIL.max_delta
        
        gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
        self.img_h, self.img_w = gray.shape
        self.model = OnlineMILBoost(w, h, cfg)
        ii = cv2.integral(gray)
        pos = self._sample_pos(self.loc)
        neg = self._sample_neg(self.loc)
        self.model.train_frame(ii, pos, neg)

    def _sample_pos(self, center):
        """
        Args:
            center: center coordinates of the bounding box
        """
        cx, cy = center
        hw, hh = self.w // 2, self.h // 2
        pts = []
        for dx in range(-self.pos_rad, self.pos_rad + 1):
            for dy in range(-self.pos_rad, self.pos_rad + 1):
                if dx*dx + dy*dy <= self.pos_rad*self.pos_rad:          # if the point is within the radius
                    x_c, y_c = cx + dx, cy + dy             # new center coordinates
                    x0, y0 = x_c - hw, y_c - hh             # top-left corner of thenew bounding box
                    if 0 <= x0 <= self.img_w - self.w and 0 <= y0 <= self.img_h - self.h:
                        pts.append((x_c, y_c))
        return pts

    def _sample_neg(self, center):
        """
        Args:
            center: center coordinates of the bounding box
        """
        cx, cy = center
        hw, hh = self.w // 2, self.h // 2
        pts = []
        while len(pts) < self.num_neg:
            dx = random.randint(-self.neg_rad, self.neg_rad)
            dy = random.randint(-self.neg_rad, self.neg_rad)
            dist2 = dx*dx + dy*dy
            if self.pos_rad*self.pos_rad < dist2 <= self.neg_rad*self.neg_rad: # exclude points within the positive radius
                x_c, y_c = cx + dx, cy + dy
                x0, y0 = x_c - hw, y_c - hh
                if 0 <= x0 <= self.img_w - self.w and 0 <= y0 <= self.img_h - self.h:
                    pts.append((x_c, y_c))
        return pts

    def process_frame(self, frame):
        """
        Args:
            frame: current frame of the video
        Returns:
            loc: updated location of the bounding box
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        H_img, W_img = gray.shape
        cx, cy = self.loc

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
            w_s = int(self.w * s)
            h_s = int(self.h * s)
            for dx in range(-self.s_rad, self.s_rad + 1, 4):
                for dy in range(-self.s_rad, self.s_rad + 1, 4):
                    x_c = cx + dx
                    y_c = cy + dy
                    # boundary check 
                    if not (0 <= x_c - w_s//2 and x_c + w_s//2 <= W_img):
                        continue
                    if not (0 <= y_c - h_s//2 and y_c + h_s//2 <= H_img):
                        continue
                    for theta in angles:
                        tasks.append((gray, x_c, y_c,
                                    w_s, h_s, theta,
                                    self.model))

        n = min(self.sample, len(tasks))
        tasks = random.sample(tasks, n)

        # 2. Define worker function
        def eval_patch(args):
            gray_f, x_c, y_c, w_s, h_s, theta, model = args
            # Rotation matrix
            M      = cv2.getRotationMatrix2D((x_c, y_c), theta, 1.0)
            # Warp the image
            warped = cv2.warpAffine(
                gray_f, M,
                (W_img, H_img),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REPLICATE
            )
            # Compute integral image
            ii    = cv2.integral(warped)
            score = model.score_patch(ii, x_c, y_c,
                                    win_w=w_s, win_h=h_s)
            return score, (x_c, y_c, w_s, h_s, theta)
        
        # 3. Run (in parallel)
        best_score, best_params = -np.inf, None
        for t in tasks:
            score, params = eval_patch(t)
            if score > best_score:
                best_score, best_params = score, params

        # 4. Pick the best
        x_c, y_c, w_c, h_c, theta = best_params
        
        # 5. Update status and train on new patch
        self.loc   = (x_c, y_c)
        self.w, self.h, self.angle = w_c, h_c, theta

        pos = self._sample_pos(self.loc)
        neg = self._sample_neg(self.loc)
        ii_frame = cv2.integral(gray)
        self.model.train_frame(ii_frame, pos, neg,
                            win_w=self.w, win_h=self.h)

        return self.loc[0], self.loc[1], self.w, self.h, self.angle
        