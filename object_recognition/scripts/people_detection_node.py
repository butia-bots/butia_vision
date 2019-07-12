#!/usr/bin/env python

import rospy
import cv2
import imutils
from imutils.object_detection import non_max_suppression
import numpy as np
from cv_bridge import CvBridge

from butia_vision_msgs.msg import Recognitions, Description
from sensor_msgs.msg import Image

BRIDGE = CvBridge()

image_subscriber = None

recognition_publisher = None

view_publisher = None

pub_msg = Recognitions()

def image2Recognitions(image_msg):
    image = BRIDGE.imgmsg_to_cv2(image_msg, desired_encoding="bgr8")

    (rects, weights) = hog.detectMultiScale(image, winStride=(8, 8),
        padding=(32, 32), scale=1.05)

    rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
    pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)

    pub_msg.header.stamp = rospy.get_rostime().now()
    pub_msg.image_header = image_msg.header

    for (xA, yA, xB, yB) in pick:
        description = Description()
        description.label_class = 'person'
        description.probability = 1.0
        description.bounding_box.minX = xA
        description.bounding_box.minY = yA
        description.bounding_box.width = xB - xA
        description.bounding_box.height = yB - yA

        pub_msg.descriptions.append(description)
    
    return pub_msg      

def recognitions2ViewImage(image_msg, recognitions_msg):
    cv_image = BRIDGE.imgmsg_to_cv2(image_msg, desired_encoding = 'rgb8')

    faces_description = []

    if(recognitions_msg != None):
        faces_description = recognitions_msg.descriptions

    for fd in faces_description:
        bb = fd.bounding_box
        label = fd.label_class + ' - ' + ('%.2f' % fd.probability)
        color = ((1-fd.probability)*255, fd.probability*255, 0)
        cv2.rectangle(cv_image, (bb.minX, bb.minY), (bb.minX + bb.width, bb.minY + bb.height), color, 2)
        cv2.putText(cv_image, label, (bb.minX, bb.minY + bb.height + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    image_view_msg = BRIDGE.cv2_to_imgmsg(cv_image, encoding = 'rgb8')
    return image_view_msg

def imageCallback(image_msg):
    pub_msg = image2Recognitions(image_msg)
    if pub_msg != None:
        recognition_publisher.publish(pub_msg)

    pub_image_msg = recognitions2ViewImage(image_msg, pub_msg)
    view_publisher.publish(pub_image_msg)  
    


if __name__ == '__main__':
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    rospy.init_node('people_detection_node', anonymous = True)

    image_subscriber = rospy.Subscriber('/butia_vision/vsb/image_rgb_raw', Image, imageCallback, queue_size=1, buff_size=2**24)

    recognition_publisher = rospy.Publisher('/butia_vision/or/people_detection', Recognitions, queue_size=1)

    view_publisher = rospy.Publisher('/butia_vision/or/people_detection_view', Image, queue_size=1)

    rospy.spin()
