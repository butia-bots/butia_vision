import cv2
import rospy
import os
import rospkg

from ShelfDetection import ShelfDetection as sd

from cv_bridge import CvBridge

from butia_vision_msgs.msg import Recognitions, Description, BoundingBox, RGBDImage, Shelf, Level, Line
from std_msgs.msg import Header
from sensor_msgs.msg import Image
from butia_vision_msgs.srv import ImageRequest

BRIDGE = CvBridge()

shelf_subscriber=None
shelf_publisher=None

image_request_client=None

def debug(image):
    shelf_view_publisher(BRIDGE.cv2_to_imgmsg(image,encoding="bgr8"))

def shelfDetectionCallBack(recognition):
    frame_id = recognition.image_header.seq
    req = image_request_client(frame_id)
    frame = req.rgbd_image.rgb
    descriptions = recognition.descriptions
    cv_frame = BRIDGE.imgmsg_to_cv2(frame, desired_encoding = 'rgb8')
    frame_rgb = cv2.cvtColor(cv_frame,cv2.COLOR_BGR2RGB)

    number_levels, lines, resul = sd.findingLevels(frame_rgb)

    debug(resul)

    object_type, objects_labels = sd.findingObjects(lines,recognition.descriptions)

    response = Shelf()
    response.image_header = recognition.image_header
    response.number_levels = number_levels
    response.levels = []
    for i in range(number_levels):
        level = Level()
        level.object_type = object_type[i]
        level.objects_labels = objects_labels[i]
        line = Line()
        line.minX = lines[i][0]
        line.maxX = lines[i][1]
        line.Y  = lines[i][2]
        response.levels.append(level)
    shelf_publisher.publish(response)


if __name__ == '__main__':
    rospy.init_node('shelf_detection_node', anonymous = True)

    param_image_request_service = rospy.get_param("/services/image_server/image_request", "/butia_vision/is/image_request")

    param_object_recognition_topic = rospy.get_param("/topics/object_recognition/object_recognition", "/butia_vision/or/object_recognition")
    param_shelf_detection_topic = rospy.get_param("/topics/shelf_detection/shelf_detection", "/butia_vision/sd/shelf_detection")

    shelf_publisher = rospy.Publisher(param_shelf_detection_topic, Shelf, queue_size = 10)
    shelf_view_publisher = rospy.Publisher("/butia_vision/sd/shelf_detection_view", Image, queue_size=1)
    shelf_subscriber = rospy.Subscriber(param_object_recognition_topic, Recognitions, shelfDetectionCallBack, queue_size = 10)
    
    rospy.wait_for_service(param_image_request_service)
    try:
        image_request_client = rospy.ServiceProxy(param_image_request_service, ImageRequest)
    except rospy.ServiceException, e:
        print "Service call failed %s"%e

    rospy.spin()
