#!/usr/bin/env python

import cv2
import rospy

from cv_bridge import CvBridge

from people_tracking_ros import PeopleTracking

from butia_vision_msgs.msg import Recognition, Description, BoundingBox, RGBDImage
from std_msgs.msg import Header
from sensor_msgs.msg import Image

from butia_vision_msgs.srv import ImageRequest, SegmentationRequest, StartTracking, StopTracking

BRIDGE = CvBridge()

bounding_box_size_threshold=None
probability_threshold=None
segmentation_type=None
queue_size=None
min_hessian=None
minimal_minimal_distance=None
matches_check_factor=None
param_k=None

people_tracking=None

tracker_publisher=None
tracker_subscriber=None

start_service=None
stop_service=None

image_request_client=None 
segmentation_request_client=None

frame=None
cv_frame=None
descriptions=None

def debug(tracker, dets):
    for track in tracker.tracks:
        if not track.is_confirmed() or track.time_since_update > 1:
				continue
        
        bbox = track.to_tlbr()
        id_num= str(track.track_id)
        features = track.features
        
        cv2.rectangle(cv_frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),(255,255,255), 2)
		cv2.putText(cv_frame, str(id_num),(int(bbox[0]), int(bbox[1])),0, 5e-3 * 200, (0,255,0),2)

        for det in dets:
            bbox = det.to_tlbr()
            cv2.rectangle(cv_frame,(int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),(255,255,0), 2)
    
    cv2.imshow('cv_frame',cv_frame)
    out.write(cv_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    

#Tracking people in real time
def peopleDetectionCallBack(recognition):
    frame_id = image_header.seq
    req = image_request_client(frame_id)
    frame = req.rgbd
    cv_frame = BRIDGE.imgmsg_to_cv2(frame.rgb, desired_encoding = 'rgb8')
    people_tracking.setFrame(recognition.image_header, recognition.header, frame_id, cv_frame)

    img_size = frame.rgb.height * frame.rgb.width
    for description in descriptions:
        if(description.bounding_box.width*description.bounding_box.height < img_size*bounding_box_size_threshold or description.probability < probability_threshold):
            descriptions.remove(description)
    
    people_tracking.generateDetections(descriptions)

    tracker, detections = people_tracking.track()

    debug(people_tracking.tracker, people_tracking.dets)

    '''
    if(people_tracking.tracking) {
        if(people_tracking.inImage()) {
            tracker_publisher.publish(people_tracking.personFound())
        }
    }
    '''


def startTracking(start):
    #people_tracking.startTrack()
    return start

def stopTracking(stop):
    #people_tracking.stopTrack()
    return stop

if __name__ == '__main__':
    rospy.init_node('people_tacking_node', anonymous = True)
    
    bounding_box_size_threshold = rospy.get_param("/people_tracking/thresholds/bounding_box_size", 0.1)
    probability_threshold = rospy.get_param("/people_tracking/thresholds/probability", 0.5)
    segmentation_type = rospy.get_param("/people_tracking/segmentation/type", "median_full")
    
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
    
    people_tracking = PeopleTracking()

    rospy.spin()
    

