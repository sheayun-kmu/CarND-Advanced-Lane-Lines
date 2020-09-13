import logging
logging.basicConfig(
    level=logging.ERROR,
    format=u'%(asctime)-15s [%(name)s] %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
import os
import glob
from moviepy.editor import VideoFileClip

from lanefinder.ImgPipeline import ImgPipeline

def detection_pipeline(img, pipeline):
    warped = pipeline.preprocess(img)
    pipeline.detect_lanes(warped, True)
    annotated = pipeline.paint_drivable()
    annotated = pipeline.annotate_info(annotated)
    return annotated

def convert_video(video_input_filename):
    basename = os.path.basename(video_input_filename)
    filename, extension = os.path.splitext(basename)
    output_pathname = os.path.join(
        os.getcwd(), "output_images", filename + '_output.mp4'
    )
    pipeline = ImgPipeline()
    clip = VideoFileClip(video_input_filename)
    output_clip = clip.fl_image(
        lambda x:detection_pipeline(x, pipeline)
    )
    output_clip.write_videofile(output_pathname, audio=False)

if __name__ == '__main__':
    curpath = os.path.dirname(os.path.abspath(__file__))
    selector = os.path.join(curpath, '.', '*.mp4')
    for vid_file in glob.glob(selector):
        convert_video(vid_file)
