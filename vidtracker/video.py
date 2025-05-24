import cv2
import os
import numpy as np
import glob
from loguru import logger
import matplotlib.pyplot as plt

from vidtracker.util import (
    show_haar_feature, 
    show_weak_classifier, 
    show_online_MIL_boost,
    show_explode_and_convolve,
    show_smoothing
)
from vidtracker.mil import MILTracker
from vidtracker.dfs import DFSTracker

def process_video(cfg):
    os.makedirs(cfg.OUTPUT.PATH, exist_ok=True)
    out_vid = os.path.join(cfg.OUTPUT.PATH, cfg.OUTPUT.FILES.VIDEO)

    # show_haar_feature()
    # show_weak_classifier()
    # show_online_MIL_boost()
    # show_explode_and_convolve()
    # show_smoothing()
    
    frames = sorted(glob.glob(os.path.join(cfg.INPUT.PATH, "img*.png")))
    if not frames:
        logger.error(f"No frames found in {cfg.INPUT.PATH}")
        return
    
    first_frame = cv2.imread(frames[0])
    init_bbox = cv2.selectROI("Select Object", first_frame, showCrosshair=False, fromCenter=False)
    if cfg.TRACKER == "MIL":
        logger.info(f"Using MIL Tracker with config: {cfg.MIL}")
        tracker = MILTracker(
            first_frame=first_frame,
            init_bbox=init_bbox,
            cfg=cfg
        )
    elif cfg.TRACKER == "DFS":
        logger.info(f"Using DFS Tracker with config: {cfg.DFS}")
        tracker = DFSTracker(
            first_frame=first_frame,
            init_bbox=init_bbox,
            cfg=cfg
        )
    else:
        logger.error(f"Unsupported tracker type: {cfg.TRACKER}")
        return
    cv2.destroyWindow("Select Object")

    for i, fname in enumerate(frames[1:]):
        frame = cv2.imread(fname)
        if frame is None:
            logger.error(f"Error reading frame: {fname}")
            continue
        cx, cy, w, h, angle = tracker.process_frame(frame)
        # define the rotated rect
        rect  = ((cx, cy), (w, h), angle)
        box   = cv2.boxPoints(rect)         # 4 corners of rotated box
        box   = np.int0(box)                # integer coords
        cv2.drawContours(frame, [box], 0, (0, 255, 0), 2)

        cv2.imshow(f"{cfg.TRACK}Track", frame)
        logger.info(f"Frame {i}/{len(frames)-1}: {fname}")
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()

    return