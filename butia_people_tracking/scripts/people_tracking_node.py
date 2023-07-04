#!/usr/bin/env python3

import cv2
import rospy
import os
import rospkg
import numpy as np
import ros_numpy

from people_tracking import PeopleTracking

from butia_vision_msgs.msg import Recognitions2D, Description2D
from sensor_msgs.msg import Image
from vision_msgs.msg import BoundingBox2D

from std_srvs.srv import Empty, EmptyResponse

matching_threshold=None
max_iou_distance=None
max_age=None
n_init=None
queue_size=None

people_tracking=None

tracker_publisher=None
tracker_subscriber=None
view_publisher=None

start_service=None
stop_service=None

image_request_client=None 
segmentation_request_client=None

frame=None
descriptions=None

PACK_DIR = rospkg.RosPack().get_path('butia_people_tracking')

def debug(cv_frame, tracker, dets, person=None):
    if tracker is not None:
        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue

            bbox = track.to_tlbr()
            id_num= str(track.track_id)
            if person is not None:
                if person == track:
                    cv_frame = cv2.rectangle(cv_frame,(int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),(0,0,255), 2)
                    cv_frame = cv2.putText(cv_frame, str(id_num),(int(bbox[0]), int(bbox[1])),0, 5e-3 * 200, (0,255,0),2)
                else:
                    cv_frame = cv2.rectangle(cv_frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),(255,255,255), 2)
                    cv_frame = cv2.putText(cv_frame, str(id_num),(int(bbox[0]), int(bbox[1])),0, 5e-3 * 200, (0,255,0),2)
            else:
                cv_frame = cv2.rectangle(cv_frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),(255,255,255), 2)
                cv_frame = cv2.putText(cv_frame, str(id_num),(int(bbox[0]), int(bbox[1])),0, 5e-3 * 200, (0,255,0),2)

    for det in dets:
        bbox = det.to_tlbr()
        cv_frame = cv2.rectangle(cv_frame,(int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),(255,255,0), 2)

    view_publisher.publish(ros_numpy.msgify(Image, np.flip(cv_frame, 2), encoding = 'rgb8'))
    

#Tracking people in real time
def peopleDetectionCallBack(recognitions):
    frame = recognitions.image_rgb
    descriptions = recognitions.descriptions
    cv_frame = ros_numpy.numpify(frame)
    people_tracking.setFrame(frame.header, recognitions.header, frame.header.seq, cv_frame.copy())

    tracker, detections = people_tracking.track(descriptions)
    if tracker is not None and people_tracking.trackingPerson is not None:
        if people_tracking.trackingPerson.is_confirmed() and people_tracking.trackingPerson.time_since_update <= 1:
            response = Recognitions2D()
            response.header = recognitions.header
            response.image_rgb = recognitions.image_rgb
            response.image_depth = recognitions.image_depth
            response.camera_info = recognitions.camera_info
            response.points = recognitions.points
            response.descriptions = [Description2D()]
            response.descriptions[0].label = 'tracked_person'
            response.descriptions[0].type = Description2D.DETECTION
            bbox = BoundingBox2D()
            bbox.size_x = people_tracking.trackingPerson.to_tlwh()[2]
            bbox.size_y = people_tracking.trackingPerson.to_tlwh()[3]
            bbox.center.x = int(people_tracking.trackingPerson.to_tlwh()[0] + bbox.size_x/2)
            bbox.center.y = int(people_tracking.trackingPerson.to_tlwh()[1] + bbox.size_y/2)
            response.descriptions[0].bbox = bbox
            tracker_publisher.publish(response)
        else:
            people_tracking.reFindPerson()
    debug(cv_frame.copy(), tracker, detections, people_tracking.trackingPerson)

def startTracking(req):
    rospy.loginfo("Starting tracking")
    people_tracking.startTrack()
    return EmptyResponse()


def stopTracking(req):
    print("Stop tracking")
    people_tracking.stopTrack()
    return EmptyResponse()

if __name__ == '__main__':
    rospy.init_node('people_tacking_node', anonymous = True)
    
    matching_threshold = rospy.get_param("/people_tracking/thresholds/matching", 0.5)
    max_iou_distance = rospy.get_param("people_tracking/thresholds/max_iou_distance", 0.5)
    max_age = rospy.get_param("people_tracking/thresholds/max_age", 60)
    n_init = rospy.get_param("people_tracking/thresholds/n_init", 5)
    
    queue_size = rospy.get_param("/people_tracking/queue/size", 1)

    param_start_service = rospy.get_param("people_tracking/services/people_tracking/start_tracking", "/butia_vision/pt/start")
    param_stop_service = rospy.get_param("people_tracking/services/people_tracking/stop_tracking", "/butia_vision/pt/stop")

    param_people_detection_topic = rospy.get_param("people_tracking/topics/people_tracking/people_detection", "/butia_vision/br/recognitions2D")
    param_people_tracking_topic = rospy.get_param("people_tracking/topics/people_tracking/people_tracking", "/butia_vision/pt/people_tracking")

    tracker_publisher = rospy.Publisher(param_people_tracking_topic, Recognitions2D, queue_size = queue_size)
    view_publisher = rospy.Publisher('/butia_vision/pt/people_tracking_view', Image, queue_size = queue_size)
    tracker_subscriber = rospy.Subscriber(param_people_detection_topic, Recognitions2D, peopleDetectionCallBack, queue_size = queue_size)

    start_service = rospy.Service(param_start_service, Empty, startTracking)
    stop_service = rospy.Service(param_stop_service, Empty, stopTracking)
    
    models_dir = os.path.join(PACK_DIR, 'models')
    model_file = os.path.join(models_dir, 'mars-small128.pb')

    people_tracking = PeopleTracking(model_file, matching_threshold, max_iou_distance, max_age, n_init)

    rospy.spin()
    

