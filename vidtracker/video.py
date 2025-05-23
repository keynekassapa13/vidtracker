import cv2
import os
import numpy as np
import glob
from loguru import logger
import matplotlib.pyplot as plt

from vidtracker.util import show_haar_feature, show_weak_classifier, show_online_MIL_boost
from vidtracker.mil import MILTracker

def process_video(cfg):
    os.makedirs(cfg.OUTPUT.PATH, exist_ok=True)
    out_vid = os.path.join(cfg.OUTPUT.PATH, cfg.OUTPUT.FILES.VIDEO)

    # show_haar_feature()
    # show_weak_classifier()
    # show_online_MIL_boost()
    
    frames = sorted(glob.glob(os.path.join(cfg.INPUT.PATH, "img*.png")))
    if not frames:
        logger.error(f"No frames found in {cfg.INPUT.PATH}")
        return
    
    first_frame = cv2.imread(frames[0])
    init_bbox = cv2.selectROI("Select Object", first_frame, showCrosshair=False, fromCenter=False)
    tracker = MILTracker(
        first_frame=first_frame,
        init_bbox=init_bbox,
        cfg=cfg
    )
    cv2.destroyWindow("Select Object")

    for i, fname in enumerate(frames[1:]):
        frame = cv2.imread(fname)
        if frame is None:
            logger.error(f"Error reading frame: {fname}")
            continue
        cx, cy = tracker.process_frame(frame)
        x1 = int(cx - tracker.w  / 2)
        y1 = int(cy - tracker.h / 2)
        x2 = int(x1 + tracker.w)
        y2 = int(y1 + tracker.h)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.imshow("MILTrack", frame)
        logger.info(f"Frame {i+1}/{len(frames)-1}: {fname}")
        cv2.waitKey(1)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()

    return