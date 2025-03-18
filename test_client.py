import os
from PIL import Image as PILImage
from sensor_msgs.msg import Image
import rospy
import numpy as np
import roslib; roslib.load_manifest('amrl_msgs')
from amrl_msgs.msg import *
from amrl_msgs.srv import VideoCaptionSrv, VideoCaptionSrvRequest

def numpy_to_ros_image(np_image, encoding="rgb8"):
    """ Convert a NumPy image (OpenCV) to a ROS Image message. """
    ros_image = Image()
    ros_image.height = np_image.shape[0]
    ros_image.width = np_image.shape[1]
    ros_image.encoding = encoding  # "bgr8" for OpenCV images, "rgb8" for PIL images
    ros_image.is_bigendian = 0
    ros_image.step = np_image.shape[1] * np_image.shape[2]  # width * channels
    ros_image.data = np_image.tobytes()
    return ros_image

def load_images(image_paths):
        """Load images from file paths as PIL images."""
        images = []
        for path in image_paths:
            if os.path.exists(path):
                try:
                    img = PILImage.open(path)
                    images.append(img)
                    print(f"Loaded image: {path}")
                except Exception as e:
                    print(f"Error loading {path}: {e}")
            else:
                print(f"File not found: {path}")
        return images
    
def call_vila_service(images, prompt):
    rospy.wait_for_service('vila_video_caption')
    try:
        vila = rospy.ServiceProxy('vila_video_caption', VideoCaptionSrv)
        video = []
        for image in images:
            if image.mode == "RGBA":
                image = image.convert("RGB")
            ros_image = numpy_to_ros_image(np.array(image))
            video.append(ros_image)
        request = VideoCaptionSrvRequest()
        request.prompt = prompt
        request.video = video
        response = vila(request)
        rospy.loginfo(f"Caption: {response.caption}")
    except rospy.ServiceException as e:
        rospy.logerr(f"Service call failed: {e}")
    
if __name__ == "__main__":
    rospy.init_node("vila_client")
    
    image_files = ["demo_images/av.png"]
    images = load_images(image_files)
    prompt = "<video>\n You are a wandering around a household area. Please describe in detail what you see in the few seconds of the video. \
        Focus on objects, events/activities, people and their actions, and other notable details. \
        Provide enough detail about objects (e.g., colors, patterns, logos, or states) to ensure they can be identified through text alone. For example,  Instead of just 'a box,' describe its color, any images or logos on it, and any distinguishing marks. \
        Think step by step about these details and be very specific. \
        Describe the video directly without any introductory phrases or extra commentary."
    
    call_vila_service(images, prompt)