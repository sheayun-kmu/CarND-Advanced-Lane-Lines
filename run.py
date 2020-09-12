import logging
logging.basicConfig(
    level=logging.ERROR,
    format=u'%(asctime)-15s [%(name)s] %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
import glob
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from lanefinder.params import perspective_params
from lanefinder.ImgPipeline import ImgPipeline
from lanefinder.CamModel import CamModel
from lanefinder.Binarizer import Binarizer
from lanefinder.LaneDetector import LaneDetector

def get_calibrated_cam():
    cam = CamModel()
    cal_images = glob.glob('./camera_cal/calibration*.jpg')
    cam.calibrate(cal_images, 9, 6)
    cam.init_perspective()
    return cam

# Juxtapose two images given by parameters.
# Default color map is 'viridis' (RGB, used by mpimg)
# while grayscale images (two dimensional) are considered.
def visual_compare(img1, title1, img2, title2, fontsize=20):
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
    f.tight_layout()
    cmap1 = 'gray' if len(img1.shape) == 2 else 'viridis'
    cmap2 = 'gray' if len(img2.shape) == 2 else 'viridis'
    ax1.imshow(img1, cmap=cmap1)
    ax1.set_title(title1, fontsize=fontsize)
    ax2.imshow(img2, cmap=cmap2)
    ax2.set_title(title2, fontsize=fontsize)
    plt.subplots_adjust(left=0, right=1, top=0.9, bottom=0)
    plt.show()

def test_undistort(imgfile):
    cam = get_calibrated_cam()
    img = mpimg.imread(imgfile)
    undistorted = cam.undistort(img)
    visual_compare(img, 'Original Image', undistorted, 'Undistorted Image')

def test_warp(imgfile):
    cam = get_calibrated_cam()
    img = mpimg.imread(imgfile)
    undistorted = cam.undistort(img)
    src = np.array(
        [
            perspective_params['src']['ul'], # upper left
            perspective_params['src']['ur'], # upper right
            perspective_params['src']['lr'], # lower right
            perspective_params['src']['ll'], # lower left
        ],
        np.int32
    )
    dst = np.array(
        [
            perspective_params['dst']['ul'], # upper left
            perspective_params['dst']['ur'], # upper right
            perspective_params['dst']['lr'], # lower right
            perspective_params['dst']['ll'], # lower left
        ],
        np.int32
    )
    warped = cam.warp(undistorted)
    undistorted = cv2.polylines(undistorted, [src], True, (255, 0, 0), 2)
    warped = cv2.polylines(warped, [dst], True, (255, 0, 0), 2)
    visual_compare(undistorted, 'Undistorted Image', warped, 'Warped Image')

def test_binarizer(imgfile):
    cam = get_calibrated_cam()
    img = mpimg.imread(imgfile)
    '''
    img = cv2.imread(imgfile)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    '''
    undistorted = cam.undistort(img)
    binarizer = Binarizer()
    bin_image = binarizer.binarize(undistorted)
    visual_compare(img, 'Original Image', bin_image, 'Binarized Image')

def test_detector(imgfile):
    pipeline = ImgPipeline()
    img = mpimg.imread(imgfile)
    warped_binary = pipeline.preprocess(img)
    pipeline.detect_lanes(warped_binary)
    annotated = pipeline.paint_drivable()
    # annotated = pipeline.debug_img
    annotated = pipeline.annotate_info(annotated)
    visual_compare(pipeline.get_undistorted(), 'Undistorted Image',
                   annotated, 'Lane Detected')

def detection_pipeline(img, img_pipeline):
    w = pipeline.preprocess(img)
    pipeline.detect_lanes(w)
    annotated = pipeline.paint_drivable()
    # annotated = pipeline.debug_img
    annotated = pipeline.annotate_info(annotated)
    return annotated

if __name__ == '__main__':
    if False:
        test_img_file = './test_images/test1.jpg'
        # test_undistort(test_img_file)
        # test_binarizer(test_img_file)
        # test_warp('./test_images/straight_lines1.jpg')
        # test_warp('./test_images/straight_lines2.jpg')
        # test_warp(test_img_file)
        test_detector(test_img_file)
    else:
        import os
        from moviepy.editor import VideoFileClip
        test_video_file = './project_video.mp4'
        # test_video_file = './20.mp4'
        # test_video_file = './challenge_video.mp4'
        pipeline = ImgPipeline()
        detector_l = LaneDetector()
        detector_r = LaneDetector()
        output_pathname = os.path.join(os.getcwd(), "output.mp4")
        clip = VideoFileClip(test_video_file)
        output_clip = clip.fl_image(
            lambda x:detection_pipeline(x, pipeline)
        )
        output_clip.write_videofile(output_pathname, audio=False)
