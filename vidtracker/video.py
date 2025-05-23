import cv2
import os
import numpy as np
import glob
from loguru import logger
import matplotlib.pyplot as plt

from vidtracker.util import show_haar_feature, show_weak_classifier, show_online_MIL_boost


def process_video(cfg):
    os.makedirs(cfg.OUTPUT.PATH, exist_ok=True)
    out_vid = os.path.join(cfg.OUTPUT.PATH, cfg.OUTPUT.FILES.VIDEO)

    show_haar_feature()
    show_weak_classifier()
    show_online_MIL_boost()
    

    return