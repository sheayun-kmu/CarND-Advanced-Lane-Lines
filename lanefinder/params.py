# Parameter definitions for lane finding

import os
import glob
import cv2

# Calibration files are parameterized.
curpath = os.path.dirname(os.path.abspath(__file__))
pathsel = os.path.join(curpath, '..', 'camera_cal', 'calibration*.jpg')
calibration_filepaths = glob.glob(pathsel)

# Perspective transformation parameters
c, r = 1280, 720
h_offset = 200
perspective_params = {
    'src': {
        'ul': [593, 450],
        'ur': [687, 450],
        'll': [220, 700],
        'lr': [1090, 700],
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
}
