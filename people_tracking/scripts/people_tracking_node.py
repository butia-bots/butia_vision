#!/usr/bin/env python

import cv2
import rospy

from cv_bridge import CvBridge

from people_tracking_ros import PeopleTrackingROS

from butia_vision_msgs.msg import Recognition, Description, BoundingBox, RGBDImage
from std_msgs.msg import Header
from sensor_msgs.msg import Image

from butia_vision_msgs.srv import ImageRequest, SegmentationRequest, StartTracking, StopTracking

BRIDGE = CvBridge()
people_tracking = PeopleTrackingROS()

def peopleDetectionCallBack():
    #Tracking people in real time

def startTracking():


def stopTracking():


if __name__ == '__main__':
    rospy.init_node('people_tacking_node', anonymous = True)
    
    bounding_box_size_threshold = rospy.get_param("/people_tracking/thresholds/bounding_box_size", 0.1)
    probability_threshold = rospy.get_param("/people_tracking/thresholds/probability", 0.5)
    
    queue_size = rospy.get_param("/people_tracking/queue/size", 20)

    min_hessian = rospy.get_param("/people_tracking/match/minimal_hessian", 400)
    minimal_minimal_distance = rospy.get_param("/people_tracking/match/minimal_minimal_distance", 0.2)
    matches_check_factor = rospy.get_param("/people_tracking/match/check_factor", 0.2)
    param_k = rospy.get_param("/people_tracking/match/k", 8)

    param_detector_type = rospy.get_param("/people_tracking/detector/type", "surf")

    param_start_service = rospy.get_param("/services/people_tracking/start_tracking", "/butia_vision/pt/start")
    param_stop_service = rospy.get_param("/services/people_tracking/stop_tracking", "/butia_vision/pt/stop")
    param_image_request_service = rospy.get_param("/services/image_server/image_request", "/butia_vision/is/image_request")
    param_segmentation_request_service = rospy.get_param("/services/segmentation/segmentation_request", "/butia_vision/seg/image_segmentation")

    param_people_detection_topic = rospy.get_param("/topics/object_recognition/people_detection", "/butia_vision/or/people_detection")
    param_people_tracking_topic = rospy.get_param("/topics/people_tracking/people_tracking", "/butia_vision/pt/people_tracking")

    tracker_publisher = rospy.Publisher(param_people_tracking_topic, Recognition, queue_size = 10)
    tracker_subscriber = rospy.Subscriber(param_people_detection_topic, Recognition, peopleDetectionCallBack, queue_size = 10)

    start_service = rospy.Service(param_start_service, StartTracking, startTracking)
    stop_service = rospy.Service(param_stop_service, StopTracking, stopTracking)
    
    rospy.wait_for_service(param_image_request_service)
    
    try:
        image_request_client = rospy.ServiceProxy(param_image_request_service, ImageRequest)
    except rospy.ServiceException, e:
        print "Service call failed %s"%e

    rospy.wait_for_service(param_segmentation_request_service)

    try:
        segmentation_request_client = rospy.ServiceProxy(param_segmentation_request_service, SegmentationRequest)
    except rospy.ServiceException, f:
        print "Service call failed %s"%e 
    
    rospy.spin()
    

