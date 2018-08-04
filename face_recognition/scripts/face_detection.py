#!/usr/bin/env python

import rospy
from openface_ros import OpenfaceROS
from sensor_msgs.msg import Image

def imageListener(image_msg):
    face_bbs = openface.getAllFaceBoundingBoxes(image_msg)
    print face_bbs
    #publishDetection(face_bbs)

def publishDetection(face_bbs):
    pass

openface = OpenfaceROS()
openface.createDlibAlign()

if __name__ == '__main__':
    rospy.init_node('face_detection_node', anonymous = True)

    rospy.Subscriber('/cv_camera/image_raw', Image, imageListener)

    #publisher = rospy.Publisher('/vision_system/fr/face_detection', DetectedFaces, queue_size = 10)

    rospy.spin()