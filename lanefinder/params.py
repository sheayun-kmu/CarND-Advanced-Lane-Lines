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

# Binarizer parameters (thresholds).
binarizer_params = {
    's_channel_color_threshold': (170, 255),
    'x_gradient_threshold': (20, 100),
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

# Conversion into real-world measurement
conversion_params = {
    'meters_per_pixel_x': 3.7 / 835,
    'meters_per_pixel_y': 30 / 720,
}

# Lande detection parameters
px_per_meter = 1 / conversion_params['meters_per_pixel_x']
detector_params = {
    'sliding_window_params': {
        'nwindows': 15,
        'margin': 100,
        'minpix': 50,
    },
    'base_drift_limit': 100,
    'parallel_check_limit': 1.0e-03,
    'curvature_diff_limit': 2.0e-03,
    'lane_width_lower_bound': 3.4 * px_per_meter,
    'lane_width_upper_bound': 4.0 * px_per_meter,
    'failure_acc_limit': 10,
    'number_of_fit_records': 5,
    'fit_avg_weights': [0.03, 0.07, 0.2, 0.3, 0.4],
    'sanity_check_params': {
        'sanity_check_base_drift': True,
        'sanity_check_curvature_diff': True,
        'sanity_check_lane_width': True,
        'sanity_check_parallel': True,
    },
}

# Display paramters
display_params = {
    'text_font': cv2.FONT_HERSHEY_SIMPLEX,
    'text_position': (50, 50),
    'font_scale': 1,
    'font_color': (255, 255, 255),
    'line_type': 2,
}
