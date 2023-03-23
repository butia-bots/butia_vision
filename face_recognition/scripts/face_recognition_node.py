#!/usr/bin/env python

import cv2
import rospy

from std_srvs.srv import Empty, EmptyResponse

from cv_bridge import CvBridge
from face_recognition_ros import FaceRecognitionROS

from std_msgs.msg import Header

from sensor_msgs.msg import Image

from butia_vision_msgs.msg import Recognitions, ClassifierReload

from butia_vision_msgs.srv import ListClasses, ListClassesResponse

BRIDGE = CvBridge()

face_recognition_ros = FaceRecognitionROS()

image_subscriber = None
reload_subscriber = None

recognition_publisher = None

face_list_updated_publisher = None

face_list_header = Header()

view_publisher = None

list_faces_server = None

def imageCallback(image_msg):
    pub_msg = face_recognition_ros.recognitionProcess(image_msg)
    if pub_msg != None:
        recognition_publisher.publish(pub_msg)

    pub_image_msg = recognizedFaces2ViewImage(image_msg, pub_msg)
    print(pub_image_msg)
    view_publisher.publish(pub_image_msg)   
    
def classifierReloadCallback(ros_msg):
    face_recognition_ros.loadClassifier(ros_msg.model_name)
    face_list_header.seq += 1 
    face_list_header.stamp = rospy.get_rostime()
    face_list_updated_publisher.publish(face_list_header)

def recognizedFaces2ViewImage(image_msg, recognized_faces_msg):
    cv_image = BRIDGE.imgmsg_to_cv2(image_msg, desired_encoding = 'rgb8')

    faces_description = []

    if(recognized_faces_msg != None):
        faces_description = recognized_faces_msg.descriptions

    for fd in faces_description:
        bb = fd.bounding_box
        label = fd.label_class + ' - ' + ('%.2f' % fd.probability)
        color = ((1-fd.probability)*255, fd.probability*255, 0)
        cv2.rectangle(cv_image, (bb.minX, bb.minY), (bb.minX + bb.width, bb.minY + bb.height), color, 2)
        cv2.putText(cv_image, label, (bb.minX, bb.minY + bb.height + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    image_view_msg = BRIDGE.cv2_to_imgmsg(cv_image, encoding = 'rgb8')
    return image_view_msg

def getFacesList(req):
    list_faces = face_recognition_ros.face_classifier[0].classes_.tolist()
    return ListClassesResponse(list_faces)

def start(req):
    global state
    state = True
    return EmptyResponse()

def stop(req):
    global state
    state = False
    return EmptyResponse()

if __name__ == '__main__':
    rospy.init_node('face_recognition_node', anonymous = True)

    camera_read_topic = rospy.get_param("/face_recognition/subscribers/camera_reading/topic", "/usb_cam/image_raw")
    camera_read_qs = rospy.get_param("/face_recognition/subscribers/camera_reading/queue_size", 1)

    classifier_reload_topic = rospy.get_param("/face_recognition/subscribers/classifier_reload/topic", "/butia_vision/fr/classifier_reload")
    classifier_reload_qs = rospy.get_param("/face_recognition/subscribers/classifier_reload/queue_size", 1)

    face_recognition_topic = rospy.get_param("/face_recognition/publishers/face_recognition/topic", "/butia_vision/fr/face_recognition")
    face_recognition_qs = rospy.get_param("/face_recognition/publishers/face_recognition/queue_size", 1)
    
    face_recognition3d_topic = rospy.get_param("/face_recognition/publishers/face_recognition3d/topic", "/butia_vision/fr/face_recognition3d")
    face_recognition3d_qs = rospy.get_param("/face_recognition/publishers/face_recognition3d/queue_size", 1)

    face_recognition_view_topic = rospy.get_param("/face_recognition/publishers/face_recognition_view/topic", "/butia_vision/fr/face_recognition_view")
    face_recognition_view_qs = rospy.get_param("/face_recognition/publishers/face_recognition_view/queue_size", 1)

    face_list_updated_topic = rospy.get_param("/face_recognition/publishers/face_list_updated/topic", "/butia_vision/fr/face_list_updated")
    face_list_updated_qs = rospy.get_param("/face_recognition/publishers/face_list_updated/queue_size", 1)

    list_faces_service = rospy.get_param("/face_recognition/servers/list_faces/service", "/butia_vision/fr/list_faces")

    start_service = rospy.get_param("/face_recognition/servers/start/service", "/butia_vision/fr/start")
    
    stop_service = rospy.get_param("/face_recognition/servers/stop/service", "/butia_vision/fr/stop")

    image_subscriber = rospy.Subscriber(camera_read_topic, Image, imageCallback, queue_size=camera_read_qs, buff_size=2**24)

    reload_subscriber = rospy.Subscriber(classifier_reload_topic, ClassifierReload, classifierReloadCallback, queue_size=classifier_reload_qs)

    recognition_publisher = rospy.Publisher(face_recognition_topic, Recognitions, queue_size=face_recognition_qs)

    view_publisher = rospy.Publisher(face_recognition_view_topic, Image, queue_size=face_recognition_view_qs)

    face_list_updated_publisher = rospy.Publisher(face_list_updated_topic, Header, queue_size=face_list_updated_topic)

    list_faces_server = rospy.Service(list_faces_service, ListClasses, getFacesList)

    start_server = rospy.Service(start_service, Empty, start)

    stop_server = rospy.Service(stop_service, Empty, stop)

    state = False

    face_list_header.stamp = rospy.get_rostime()
    face_list_header.frame_id = "list_faces"

    face_list_updated_publisher.publish(face_list_header)

    rospy.spin()
