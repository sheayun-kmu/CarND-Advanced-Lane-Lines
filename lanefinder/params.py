# Parameter definitions for lane finding

import cv2

# Perspective transformation parameters
c, r = 1280, 720
h_offset = 300
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
