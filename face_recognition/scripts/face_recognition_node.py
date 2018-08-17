#!/usr/bin/env python

import cv2
import rospy

from cv_bridge import CvBridge
from openface_ros import OpenfaceROS
from sensor_msgs.msg import Image
from vision_system_msgs.msg import RecognizedFaces, ClassifierReload

BRIDGE = CvBridge()

def recognizedFaces2ViewImage(image_msg, recognized_faces_msg):
    #image_view_msg = image_msg
    #image_view_msg.header.stamp = rospy.get_rostime()

    cv_image = BRIDGE.imgmsg_to_cv2(image_msg, desired_encoding = 'passthrough')

    faces_description = []

    if(recognized_faces_msg != None):
        faces_description = recognized_faces_msg.faces_description

    for fd in faces_description:
        bb = fd.bounding_box
        label = fd.label_class + ' - ' + ('%.2f' % fd.probability)
        cv2.rectangle(cv_image, (bb.minX, bb.minY), (bb.minX + bb.width, bb.minY + bb.height), (255, 0, 0), 2)
        cv2.putText(cv_image, label, (bb.minX, bb.minY + bb.height + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    
    image_view_msg = BRIDGE.cv2_to_imgmsg(cv_image, encoding = 'rgb8')
    return image_view_msg

def imageListener(image_msg):
    pub_msg = openface.recognitionProcess(image_msg)

    if pub_msg != None:
        publisher.publish(pub_msg)
        
    pub_image_msg = recognizedFaces2ViewImage(image_msg, pub_msg)
    view_publisher.publish(pub_image_msg)

def classifierReload(ros_msg):
    print('Loading ' + ros_msg.classifier_name + '.')
    openface.createClassifier(ros_msg.classifier_name)
    print('Loaded.')

openface = OpenfaceROS()

publisher = None
view_publisher = None

if __name__ == '__main__':
    rospy.init_node('face_recognition_node', anonymous = True)

    rospy.Subscriber('/usb_cam/image_raw', Image, imageListener)

    rospy.Subscriber('/vision_system/fr/classifier_reload', ClassifierReload, classifierReload)

    publisher = rospy.Publisher('/vision_system/fr/face_recognition', RecognizedFaces, queue_size = 100)

    view_publisher = rospy.Publisher('/vision_system/fr/face_recognition_view', Image, queue_size = 100)

    rospy.spin()