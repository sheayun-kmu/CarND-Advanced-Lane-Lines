## Advanced Lane Finding

[//]: # (Image References)

[undistorted]: ./doc_images/undistorted.png
[warped]: ./doc_images/warped.png
[binarized]: ./doc_images/binarized.png
[detected]: ./doc_images/detected.png
[detected_debug]: ./doc_images/detected_debug.png

### Goals of the project:


1. Detect lane lines in a still image.
2. Detect lane lines in a video, which is a sequence of still images. However, unlike the above, apply heuristics to search for lane lines based on previously detected ones.

---

### Rubric Points

#### Camera Calibration

Images contained in directory `camera_cal` are used for calibrating the camera. Chessboard corners are identified (where three out of a total of 20 images are rejected because they do not contain 9 x 6 corners like others), and the corresponding grid points are mapped. Using `cv2`'s `clibrateCamera()` function, transformation parameters are obtained. The following figure shows the calibration result obtained from `camera_cal/caslibration1.jpg`.

![undistorted.png][undistorted]

#### Pipeline (single images)

Given a still image, it is preprocessed for lane detection. The steps are: (1) distortion correction, (2) binarization (color & gradient thresholding), and (3) warp to a top-down (bird's eye view) perspective. 

First, the undistorted image is binarized using color and gradient thresholding. The original image is converted into the HLS color space, where the S channel is extracted is used for color thresholding and the L channel for calculating gradient along X-axis is computed and then thresholded. With these two combined, the resulting binary image has pixels in areas that are likely to have lane lines (of course with noises). The result from binarizing `test_images/test1.jpg` is shown below:

![binarized.png][binarized]

For the perspective transform, two straight-line images (contained in `test_images`) are used, where drawing of a trapezoid is repeated for both images and the coordinates of four corner points are recorded in the pipeline's parameter set. The result (used for the purpose of basic verification by human eyes) from transforming `test_images/test1.jpg` is shown below (depicted in color for clear visualization, although the perspective transformation follows binarization described above):

![warped.png][warped]

From the binarized and then warped image, two (probable) lane lines are detected based on the sliding window algorithm: Beginning from the histogram peaks (the X-axis coordinate where the most candidate lane line pixels are located), a sliding window is moved along the candidate pixel distribution, collecting (assumed to be) valid lane line pixels. The curvature of each lane line is fit to such pixels. The radius of the curvature is calculated for future use in determining the vehicle's trajectory.

For the purpose of monitoring and debugging, the undistorted image is annotated by overlaying lane line detection information. First, the area between left and right lines are painted green, then candidate pixels are sprinkled with red (left) and blue (right) pixels. Second, textual information (radius of curvature, vehicle's relative position, and lane width) is displayed on the image. The result from annotating `test_images/test1.jpg` is shown below:

![detected.png][detected]

While this image is used for displaying the detection results for test video files, sometimes a comprehensive image for lane detection process is needed. To this end, an extra image for debugging purposes is taken (kept inside the software) where lane line pixels determined by the algorithm is visualized and the parabola fit is depicted in green, such as (for `test_images/test1.jpg`):

![detected_debug.png][detected_debug]

- Pipeline (video)

In applying lane line detection algorithm to a video file, the same approach is used as in detecting lane lines from a still image. However, one important difference is that we incorporate a separate algorithm that searches for lane line pixels around the previously fit polynomial, since we have a sequence of gradually changing images.

However, in detecting moving (though actually what moves is the camera, not the lane lines) lane lines, we might lose track of one or both of the left and right lane lines. Therefore, we perform the following sanity checks:

	1. whether the detected line is reasonably close to the previously detected line,
	2. whether the curvature of each line abruptly jumps from the previous one by a certain threshold,
	3. whether the distance between the left and the right lane lines (the lane width) is within a reasonable bound, and
	4. whether the two detected lane lines are approximately parallel.

If any of the above check fails for a certain number of consecutive frames, we reset the current guess of trying to find the lane line pixels around the previously fit polynomial and resort to the sliding window algorithm that begins from the histogram peaks.

The three test input videos are fed to the pipeline, and the output video for each of them is saved under `output_images` with respective filenames appended by `_output`.

---

### Implementation Details

The lane line detection software is written in Python, which is first tested and debugged/tuned on a local computer with all the required library packages are installed. Then it is tested on a local Jupyter server, after which all the source files and the notebook itself are uploaded onto the Udacity's Jupyter Notebook server. Again, for verification purposes, the notebook is run on the server and results are collected.

The source code is partitioned into several modules, which are stored in the Python package (and thus directory) `lanefinder`:

* `params.py` - collection of parameters used in lane detection and visualization
* `ImgPipeline.py` - main module for processing the image and detecting lane lines
* `LaneLine.py` - abstracted class representation of lane line detection status (one for each of left and right lane lines is instantiated)
* `LaneDetector.py` - a detector class that implements (1) the sliding window algorithm, and (2) the search for lane line pixels around the previously fit polynomial
* `CamModel.py` - camera model that captures the distortion correction matrix and perspective transforms (warp and unwarp)
* `Binarizer.py` - binarizer that performs color and gradient thresholding

Besides, two driver scripts are written:

* `test.py` - test functions for each stage of the pipeline; the notebook runs each step and produces visual output for monitoring purposes.
* `go.py` - run the lane line detection pipeline for each of the input video files (`project_video.mp4`, `challenge_video.mp4`, `harder_challenge_video.mp4`); not used by the notebook.

---

### Discussion

For basic test image (`test_images/test1.jpg`), the lane line detector works well. For debugging and testing purposes, several other images have been taken from the video files (which are not included in the repository), and results on some were successful and on others were not. Tuning the binary thresholding parameters is not a simple job.

Radius of curvature, vehicle position, and lane width are calculated using a very rough approximation of "meters per pixel" calculation by a rule of thumb. Fine-tuning the parameters should not be a big deal, though (a matter of specifying a value as accurately calculated as possible - not affecting any other part of the algorithm).

For most of the part of `project_video.mp4`, the lane lines were successfully detected. Sanity checks seldom failed, depending on the parameters (specifed in `params.py`). Lighting condition does not change much and we do not have any steep turns (probably recorded on a highway).

For `challenge_video.mp4`, the algorithm is often confused by something like overpavement(?) in the lane. Along this crack(?), the surface of either side has very different color which makes the x-gradient high, resulting in a mistakenly detected lane line. Another difficult thing to handle is the shadow on the left-hand side with high contrast. Nonetheless, the painted area in the output video does not go astry too much since this road is nearly straight (also probably recorded on a highway).

However, for `harder_challenge_video.mp4`, the algorithm performs quite badly. Lighting condition change a lot (mainly due to the trees alongside the road and strong sunlight making high contrast), and turns are steep. Especially, when the car enters a very curvy segment, the camera loses sight of the right lane line entirely, where this algorithm does not know what to do. For possible remedies, the following issues could be addressed:

* Fine-tune binary thresholding - do not rely on fixed threshold paramters; use adaptive thresholding paramters depending on the current image status.
* More precise determination of lane line pixels and non-lane line pixels - cleverer heuristics to discriminate pixels that are part of the lines and those that are not; for example, take into account the fact that lane lines do not change direction within a very short span.
* More robust detection algorithm - if one of the lane lines are out of the field of view, for example, determine which of the left and right lane line should be visible and adjust the detection algorithm accordingly.
