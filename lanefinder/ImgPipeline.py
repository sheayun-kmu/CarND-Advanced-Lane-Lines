import logging
import numpy as np
import cv2

from lanefinder.params import calibration_filepaths
from lanefinder.params import perspective_params
from lanefinder.CamModel import CamModel
from lanefinder.Binarizer import Binarizer

# Image processing pipeline
# 1. initialize
#    - camera calibration
#    - perspective transformation parameters
# 2. pipelining
#    - input: image taken by the camera (possibly from video)
#    - output: undistorted, binarized, warped image

class ImgPipeline:

    # Calibrate camera & setup perspective transformation parameters.
    def __init__(self, calib_image_files=None):
        self.log = logging.getLogger(__name__)
        self.log.setLevel(logging.DEBUG)
        # Get a calibrated camera model for distortion correction.
        self.cam = CamModel()
        if not calib_image_files:
            calib_image_files = calibration_filepaths
        # If we still don't have no calibration image file, it's an error.
        if not calib_image_files:
            self.log.error("No calibration image found")
            import sys
            sys.exit(-1)
        self.cam.calibrate(calib_image_files, 9, 6)
        # Initialize the camera's perspective transform.
        self.cam.init_perspective()
        # Initialize a binarizer (with default thresholds).
        self.bin = Binarizer()

    def process(self, img):
        undistorted = self.cam.undistort(img)
        binarized = self.bin.binarize(undistorted)
        recolorized = np.dstack((binarized, binarized, binarized)) * 255
        result = self.cam.warp(binarized)

        return result
