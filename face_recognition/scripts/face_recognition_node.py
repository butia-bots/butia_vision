#!/usr/bin/env python

import rospy

from openface_ros import OpenfaceROS
from sensor_msgs.msg import Image
from vision_system_msgs.msg import RecognizedFaces

def imageListener(image_msg):
    pub_msg = openface.recognitionProcess(image_msg)

    if pub_msg != None:
        publisher.publish(pub_msg)


openface = OpenfaceROS()
openface.createDlibAlign()
openface.createTorchNeuralNet()

publisher = None

if __name__ == '__main__':
    rospy.init_node('face_recognition_node', anonymous = True)

    rospy.Subscriber('/usb_cam/image_raw', Image, imageListener)

    publisher = rospy.Publisher('/vision_system/fr/face_recognition', RecognizedFaces, queue_size = 100)

    rospy.spin()