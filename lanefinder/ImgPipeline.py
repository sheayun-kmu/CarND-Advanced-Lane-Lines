import logging
import glob
import numpy as np
import cv2

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
    def __init__(self, calib_img_path=None):
        self.log = logging.getLogger(__name__)
        self.log.setLevel(logging.DEBUG)
        # Get a calibrated camera model for distortion correction.
        self.cam = CamModel()
        if calib_img_path:
            pathsel = calib_img_path
        else:
            import os
            pathsel = os.path.join(
                os.getcwd(), 'camera_cal', 'calibration*.jpg'
            )
        cal_images = glob.glob(pathsel)
        if len(cal_images) <= 0:
            self.log.error("No calibration image found by %s" % pathsel)
            import sys
            sys.exit(-1)
        self.cam.calibrate(cal_images, 9, 6)
        # Initialize the camera's perspective transform.
        self.cam.init_perspective()
        # Initialize a binarizer (with default thresholds).
        self.bin = Binarizer()

    def process(self, img):
        undistorted = self.cam.undistort(img)
        binarized = self.bin.binarize(undistorted)
        recolorized = np.dstack((binarized, binarized, binarized)) * 255
        result = self.cam.warp(recolorized)

        return result
