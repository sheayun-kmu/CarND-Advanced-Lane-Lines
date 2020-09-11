import logging
import numpy as np
import cv2

from lanefinder.params import perspective_params

# Capture camera model
# 1. calibrate using a set of images of chessboards
# 2. undistort image based on calibration
# 3. warp image to get a bird's eye view of it

class CamModel:

    # Configure logger, initialize distortion parameters
    # and perspective transform parameters.
    def __init__(self):
        self.log = logging.getLogger(__name__)
        self.log.setLevel(logging.WARN)
        self.mtx = None
        self.dist = None
        self.M = None
        self.Minv = None

    # Given a set of chessboard images (and # of corners),
    # calibrate the camera and derive the conversion matrix.
    def calibrate(self, image_files, nx, ny):
        # Initialize empty imgpoints and objpoints
        imgpoints = []
        objpoints = []
        # Prepare uniform object points using simple arithmetic
        objp = np.zeros((ny * nx, 3), np.float32)
        objp[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)

        self.log.info("Beginning camera calibration with"
                      " %d images" % len(image_files))
        for fname in image_files:
            img = cv2.imread(fname)
            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # Find chessboard corners
            ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
            # If found, add object points and image points
            if ret == True:
                imgpoints.append(corners)
                objpoints.append(objp)
            else:
                self.log.warn("Failed to find %d * %d corners from"
                              " image %s" % (nx, ny, fname))
        # Calibrate
        self.log.info("Gathered %d sets of corners" % len(imgpoints))
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
            objpoints, imgpoints, img.shape[1::-1], None, None
        )
        self.log.info("Finished calibrating camera")
        self.mtx = mtx
        self.dist = dist

    # Given an image, return its undistorted version, based on
    # calibration parameters determined in calibrate()
    def undistort(self, img):
        undist = cv2.undistort(img, self.mtx, self.dist, None, self.mtx)
        return undist

    # Given two sets of four points (corners) - src & dst -
    # compute perspective transformation (matrix M) and its inverse.
    def init_perspective(self, src=None, dst=None):
        if not src:
            src = np.float32([
                perspective_params['src']['ul'],
                perspective_params['src']['ur'],
                perspective_params['src']['ll'],
                perspective_params['src']['lr'],
            ])
        if not dst:
            dst = np.float32([
                perspective_params['dst']['ul'],
                perspective_params['dst']['ur'],
                perspective_params['dst']['ll'],
                perspective_params['dst']['lr'],
            ])
        self.M = cv2.getPerspectiveTransform(src, dst)
        self.Minv = cv2.getPerspectiveTransform(dst, src)

    # Warp image using previously initialized transform.
    def warp(self, img):
        r, c = img.shape[:2]
        warped = cv2.warpPerspective(
            img, self.M, (c, r),
            flags=perspective_params['flags']
        )
        return warped

    # Inverse-warp image using previously initialized transform.
    def inverse_warp(self, img):
        r, c = img.shape[:2]
        inverse = cv2.warpPerspective(
            img, self.Minv, (c, r),
            flags=perspective_params['flags']
        )
        return inverse
