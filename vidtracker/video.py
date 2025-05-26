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
from vidtracker.lk import LKTracker

import time

def show_video(cfg):
    """
    Show the video frames with the selected tracker.
    """
    frames = sorted(glob.glob(os.path.join(cfg.OUTPUT.PATH, "img*.png")))
    if not frames:
        logger.error(f"No frames found in {cfg.OUTPUT.PATH}")
        return
    frame = cv2.imread(frames[0])
    if frame is None:
        logger.error(f"Error reading first frame: {frames[0]}")
        return
    h, w = frame.shape[:2]
    fps = cfg.OUTPUT.FPS if cfg.OUTPUT.FPS else 30
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(os.path.join(cfg.OUTPUT.PATH, "vid.mp4"), fourcc, fps, (w, h))
    for i, fname in enumerate(frames):
        frame = cv2.imread(fname)
        if frame is None:
            logger.error(f"Error reading frame: {fname}")
            continue
        writer.write(frame)
        cv2.imshow("Video", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        logger.info(f"Frame {i+1}/{len(frames)}: {fname}")
        time.sleep(1 / fps) 

    logger.success(f"Video saved to {os.path.join(cfg.OUTPUT.PATH, 'vid.mp4')}")
    writer.release()
    cv2.destroyAllWindows()

def process_video(cfg):
    os.makedirs(cfg.OUTPUT.PATH, exist_ok=True)

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
    elif cfg.TRACKER == "LK":
        logger.info(f"Using LK Tracker with config: {cfg.LK}")
        tracker = LKTracker(
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
        # cx = center x, cy = center y, w = width, h = height, angle = rotation angle
        cx, cy, w, h, angle = tracker.process_frame(frame)
        # define the rotated rect
        rect  = ((cx, cy), (w, h), angle)
        box   = cv2.boxPoints(rect)         # 4 corners of rotated box
        box   = np.int0(box)                # integer coords
        cv2.drawContours(frame, [box], 0, (0, 255, 0), 2)

        if cfg.SHOW_FRAMES:
            cv2.imshow(f"{cfg.TRACKER}Track", frame)
        logger.info(f"Frame {i}/{len(frames)-1}: {fname}")

        output_fname = os.path.join(cfg.OUTPUT.PATH, f"img{i:05d}.png")
        cv2.imwrite(output_fname, frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cv2.destroyAllWindows()
    logger.success(f"Processed {len(frames)} frames. Output saved to {cfg.OUTPUT.PATH}")

    return