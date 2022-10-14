#!/usr/bin/env python3

import cv2
import rospy
import os
import rospkg
import numpy as np
import ros_numpy
import torchreid
from torch.nn import functional as F

from butia_vision_msgs.msg import Recognitions2D, Description2D
from sensor_msgs.msg import Image
from vision_msgs.msg import BoundingBox2D

from std_srvs.srv import Empty, EmptyResponse

from people_reid import PeopleReId

PACK_DIR = rospkg.RosPack().get_path('butia_people_tracking')

def debug(cv_frame, dets):
    detections = np.array([(int(d.bbox.center.x-d.bbox.size_x/2), int(d.bbox.center.y-d.bbox.size_y/2), d.bbox.size_x, d.bbox.size_y) for d in dets])
    for bbox in detections:
        cv_frame = cv2.rectangle(cv_frame,(int(bbox[0]), int(bbox[1])), (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3])),(255,255,0), 2)
    view_publisher.publish(ros_numpy.msgify(Image, np.flip(cv_frame, 2), encoding = 'rgb8'))

def peopleDetectionCallBack(recognitions):
    frame = recognitions.image_rgb
    descriptions = recognitions.descriptions
    print('descriptions', len(descriptions))
    cv_frame = ros_numpy.numpify(frame)
    print(cv_frame.shape)
    people_reid.setFrame(frame.header, recognitions.header, frame.header.seq, cv_frame.copy())
    people_reid.setDetections(descriptions.copy())
    if people_reid.is_tracking:
        matched_descriptions = people_reid.reid()
        response = Recognitions2D()
        response.header = recognitions.header
        response.image_rgb = recognitions.image_rgb
        response.image_depth = recognitions.image_depth
        response.camera_info = recognitions.camera_info
        response.points = recognitions.points
        response.descriptions = matched_descriptions
        response.descriptions[0].label = 'tracked_person'
        tracker_publisher.publish(response)
        #debug(people_reid.person_images[0].copy(), [])
    else:
        matched_descriptions = []
    print('matched_descriptions', len(matched_descriptions))
    debug(cv_frame.copy(), matched_descriptions.copy())

def startTracking(req):
    rospy.loginfo("Starting tracking")
    people_reid.startTrack()
    return EmptyResponse()


def stopTracking(req):
    print("Stop tracking")
    people_reid.stopTrack()
    return EmptyResponse()

if __name__ == '__main__':
    rospy.init_node('people_tacking_node', anonymous = True)

    people_reid = PeopleReId()

    
    matching_threshold = rospy.get_param("/people_tracking/thresholds/matching", 0.5)
    max_iou_distance = rospy.get_param("people_tracking/thresholds/max_iou_distance", 0.5)
    max_age = rospy.get_param("people_tracking/thresholds/max_age", 60)
    n_init = rospy.get_param("people_tracking/thresholds/n_init", 5)
    
    queue_size = rospy.get_param("/people_tracking/queue/size", 1)

    param_start_service = rospy.get_param("/services/people_tracking/start_tracking", "/butia_vision/pt/start")
    param_stop_service = rospy.get_param("/services/people_tracking/stop_tracking", "/butia_vision/pt/stop")

    param_people_detection_topic = rospy.get_param("/topics/butia_recognition/people_detection", "/butia_vision/br/people_detection")
    param_people_tracking_topic = rospy.get_param("/topics/people_tracking/people_tracking", "/butia_vision/pt/people_tracking")

    tracker_publisher = rospy.Publisher(param_people_tracking_topic, Recognitions2D, queue_size = queue_size)
    view_publisher = rospy.Publisher('/butia_vision/pt/people_tracking_view', Image, queue_size = queue_size)
    tracker_subscriber = rospy.Subscriber(param_people_detection_topic, Recognitions2D, peopleDetectionCallBack, queue_size = queue_size)

    start_service = rospy.Service(param_start_service, Empty, startTracking)
    stop_service = rospy.Service(param_stop_service, Empty, stopTracking)

    rospy.spin()