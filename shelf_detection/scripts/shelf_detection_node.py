import cv2
import rospy
import os
import rospkg

from cv_bridge import CvBridge

from butia_vision_msgs.msg import Recognitions, Description, BoundingBox, RGBDImage, Shelf, Level, Line
from std_msgs.msg import Header
from sensor_msgs.msg import Image
from butia_vision_msgs.srv import ImageRequest

BRIDGE = CvBridge()

shelf_subscriber=None
shelf_publisher=None

image_request_client=None

def shelfDetectionCallBack(recognition):
    frame_id = recognition.image_header.seq
    req = image_request_client(frame_id)
    frame = req.rgbd_image.rgb
    descriptions = recognition.descriptions
    cv_frame = BRIDGE.imgmsg_to_cv2(frame, desired_encoding = 'rgb8')
    frame_rgb = cv2.cvtColor(cv_frame,cv2.COLOR_BGR2RGB)

    # Shelf detection part

if __name__ == '__main__':
    rospy.init_node('shelf_detection_node', anonymous = True)

    param_image_request_service = rospy.get_param("/services/image_server/image_request", "/butia_vision/is/image_request")

    param_object_recognition_topic = rospy.get_param("/topics/object_recognition/object_recognition", "/butia_vision/or/object_recognition")
    param_shelf_detection_topic = rospy.get_param("/topics/shelf_detection/shelf_detection", "/butia_vision/sd/shelf_detection")

    shelf_publisher = rospy.Publisher(param_shelf_detection_topic, Shelf, queue_size = 10)
    shelf_subscriber = rospy.Subscriber(param_object_recognition_topic, Recognitions, shelfDetectionCallBack, queue_size = 10)
    
    rospy.wait_for_service(param_image_request_service)
    try:
        image_request_client = rospy.ServiceProxy(param_image_request_service, ImageRequest)
    except rospy.ServiceException, e:
        print "Service call failed %s"%e

    rospy.spin()
