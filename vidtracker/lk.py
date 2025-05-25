"""
Implementation of Lucas-Kadane Object Tracker in Videos
"""

import cv2
import numpy as np
import random

class LKTracker:
    def __init__(self, first_frame, init_bbox, cfg):
        """
        Args:
            first_frame: The first frame of the video where the object is located.
            init_bbox: Initial bounding box in the format (x, y, width, height).
            cfg: Configuration object containing parameters for the tracker.
                cfg.LK.max_corners: Maximum number of corners to detect.
                cfg.LK.quality_level: Quality level for corner detection.
                cfg.LK.min_distance: Minimum distance between corners.
                cfg.LK.block_size: Size of the block for corner detection.
                cfg.LK.win_size: Size of the window for optical flow calculation.
                cfg.LK.max_level: Maximum level of pyramid for optical flow.
                cfg.LK.min_track: Minimum number of points to track.
        """
        x, y, w, h = map(int, init_bbox)
        self.bbox = np.array([x, y, w, h], dtype=float)
        self.cfg = cfg
        self.angle = 0.0
        gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
        self.prev_gray = gray
        mask = np.zeros_like(gray)
        mask[y:y+h, x:x+w] = 255
        # Shi-Tomasi corner detection
        pts = cv2.goodFeaturesToTrack(
            gray,
            maxCorners=cfg.LK.max_corners,
            qualityLevel=cfg.LK.quality_level,
            minDistance=cfg.LK.min_distance,
            blockSize=cfg.LK.block_size,
            mask=mask
        )
        if pts is None or len(pts) < cfg.LK.min_track:
            # fallback to corners of the bounding box
            corners = np.array([[x, y], [x+w, y], [x+w, y+h], [x, y+h]], dtype=np.float32)
            pts = corners.reshape(-1, 1, 2)
            
        self.prev_pts = pts.astype(np.float32)
        self.criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03) 

    def calc_optical_flow(self, gray):
        """
        Args:
            gray: current frame in grayscale
        Returns:
            prev: points from the previous frame that were tracked
            next: points in the current frame corresponding to the previous points
        """

        if self.prev_pts is None or len(self.prev_pts) == 0:
            return np.empty((0, 2)), np.empty((0, 2))
        # Lucas Kanade optical flow

        next_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            self.prev_gray, gray, self.prev_pts, None,
            winSize=self.cfg.LK.win_size,
            maxLevel=self.cfg.LK.max_level,
            criteria=self.criteria
        )
        if status is None or next_pts is None:
            # Fallback 
            return np.empty((0, 2)), np.empty((0, 2))
        
        # mask includes only points that were successfully tracked
        mask = status.squeeze() == 1
        # good points from prev frame
        prev = self.prev_pts[mask].reshape(-1, 2)
        # good points from next frame
        nxt = next_pts[mask].reshape(-1, 2)
        return prev, nxt

    def redetect_n_track(self, gray):
        """
        Re-detect new corner features
        Tracks new detected points
        Args:
            gray: current frame in grayscale
        Returns:
            prev: points from the previous frame that were tracked
            next: points in the current frame corresponding to the previous points
        """
        x, y, w, h = self.bbox
        x, y, w, h = map(int, [x, y, w, h])
        mask = np.zeros_like(gray)
        mask[y:y+h, x:x+w] = 255
        new_pts = cv2.goodFeaturesToTrack(
            self.prev_gray,
            maxCorners=self.cfg.LK.max_corners,
            qualityLevel=self.cfg.LK.quality_level,
            minDistance=self.cfg.LK.min_distance,
            blockSize=self.cfg.LK.block_size,
            mask=mask
        )
        if new_pts is None:
            return np.empty((0, 2)), np.empty((0, 2))
        next_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            self.prev_gray, gray, new_pts, None,
            winSize=self.cfg.LK.win_size,
            maxLevel=self.cfg.LK.max_level,
            criteria=self.criteria
        )
        if status is None or next_pts is None:
            return np.empty((0, 2)), np.empty((0, 2))
        mask2 = status.squeeze() == 1
        prev = new_pts[mask2].reshape(-1, 2)
        nxt = next_pts[mask2].reshape(-1, 2)
        return prev, nxt
    
    def estimate_transform(self, prev, next):
        """
        Args:
            prev: points from the previous frame that were tracked
            next: points in the current frame corresponding to the previous points
        Returns:
            M: transformation matrix (affine)
            angle: rotation angle in degrees
        """
        if len(prev) >= 3 and len(next) >= 3:
            M, _ = cv2.estimateAffinePartial2D(prev, next, method=cv2.RANSAC)
            if M is not None:
                alpha, beta = M[0, 0], M[0, 1]
                angle = np.degrees(np.arctan2(beta, alpha))
                return M, angle
            
        # fallback: translation only
        flow = next - prev
        dx, dy = np.median(flow, axis=0) if len(flow) > 0 else (0.0, 0.0)
        M = np.array([[1, 0, dx], [0, 1, dy]], dtype=np.float32)
        return M, 0.0

    def update_bbox(self, M):
        """
        Update bbox based on the transformation matrix
        Args:
            M: transformation matrix (affine)
        """
        x, y, w, h = self.bbox
        corners = np.array([[x, y], [x+w, y], [x+w, y+h], [x, y+h]], dtype=np.float32).reshape(-1, 1, 2)
        new_corners = cv2.transform(corners, M)
        xs = new_corners[:, 0, 0]
        ys = new_corners[:, 0, 1]
        x0, y0 = float(xs.min()), float(ys.min())
        w0, h0 = float(xs.max() - xs.min()), float(ys.max() - ys.min())
        self.bbox = np.array([x0, y0, w0, h0], dtype=float)


    def refresh_features(self, gray):
        """
        Refresh the features to track
        Args:
            gray: current frame in grayscale
        """
        x, y, w, h = self.bbox
        x, y, w, h = map(int, [x, y, w, h])
        mask = np.zeros_like(gray)
        mask[y:y+h, x:x+w] = 255
        pts = cv2.goodFeaturesToTrack(
            gray,
            maxCorners=self.cfg.LK.max_corners,
            qualityLevel=self.cfg.LK.quality_level,
            minDistance=self.cfg.LK.min_distance,
            blockSize=self.cfg.LK.block_size,
            mask=mask
        )
        if pts is not None:
            self.prev_pts = pts.reshape(-1, 1, 2).astype(np.float32)
        else:

            self.prev_pts = None
    
    def process_frame(self, frame):
        """
        Args:
            frame: current frame of the video
        Returns:
            loc: updated location of the bounding box
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        good_prev, good_next = self.calc_optical_flow(gray)

        if len(good_prev) < self.cfg.LK.min_track:
            good_prev, good_next = self.redetect_n_track(gray)

        M, angle = self.estimate_transform(good_prev, good_next)
        self.update_bbox(M)
        self.angle = float(angle)

        cx = float(self.bbox[0] + self.bbox[2] / 2.0)
        cy = float(self.bbox[1] + self.bbox[3] / 2.0)
        w = float(self.bbox[2])
        h = float(self.bbox[3])
        # Re-detect good points within the new bbox
        self.refresh_features(gray)
        self.prev_gray = gray 
        return cx, cy, w, h, self.angle