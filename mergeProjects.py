
import VehicleDetectionAndTracking
import advancedLaneDetect_Pipeline
from moviepy.editor import VideoFileClip


######################################################################################################################
#  Following method combines the 2 pipelines of Advanced lane detection and Vehicle tracking.
######################################################################################################################
def mergeProjectPipelines(img):
    p5img = VehicleDetectionAndTracking.detectCars(img)
    return advancedLaneDetect_Pipeline.findLaneLinesPipeline(p5img)

######################################################################################################################
#  Following section creates the video
######################################################################################################################
def createOutVideo():
    white_output = 'project_result.mp4'
    clip1 = VideoFileClip("project_video.mp4")
    white_clip = clip1.fl_image(mergeProjectPipelines)
    white_clip.write_videofile(white_output, audio=False)
    return white_clip

# Reset the data structures used in Advanced lane detection and Vehicle tracking to start with fresh data
advancedLaneDetect_Pipeline.lastFrameData.resetLaneData()
VehicleDetectionAndTracking.heatmaps.clear()
# Create Video
white_clip = createOutVideo()
