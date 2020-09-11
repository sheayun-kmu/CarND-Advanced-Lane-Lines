# Parameter definitions for lane finding

import os
import glob
import cv2

# Calibration files are parameterized.
curpath = os.path.dirname(os.path.abspath(__file__))
pathsel = os.path.join(curpath, '..', 'camera_cal', 'calibration*.jpg')
camera_params = {
    'filepaths': glob.glob(pathsel),
    'nx': 9,
    'ny': 6,
}

# Perspective transformation parameters
c, r = 1280, 720        # image x & y size
h_offset = 200          # (left & right) margin for dst
perspective_params = {
    'src': {
        'ul': [593, 450],       # upper left
        'ur': [687, 450],       # upper right
        'll': [220, 700],       # lower left
        'lr': [1090, 700],      # lower right
    },
    'dst': {
        'ul': [0 + h_offset, 0],
        'ur': [c - h_offset, 0],
        'll': [0 + h_offset, r],
        'lr': [c - h_offset, r],
    },
    'flags': cv2.INTER_NEAREST,
}

# Sliding window parameters
detector_params = {
    'sliding_window_params': {
        'nwindows': 9,
        'margin': 100,
        'minpix': 50,
    },
    'base_drift_limit': 100,
    'parallel_check_limit': 1e-03,
}
