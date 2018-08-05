#!/usr/bin/env python

import rospy

from openface_ros import OpenfaceROS
from sensor_msgs.msg import Image
from vision_system_msgs.msg import DetectedFaces

def imageListener(image_msg):
    pub_msg = DetectedFaces()

    header = image_msg.header
    face_bbs = openface.getAllFaceBoundingBoxes(image_msg)
    header.stamp = rospy.get_rostime()

    bounding_boxes = []
    for bb in face_bbs:
        bounding_box = BoundingBox()
        bounding_box.minX = bb[0]
        bounding_box.minY = bb[1]
        bounding_box.width = bb[2]
        bounding_box.height = bb[3]
        bounding_boxes.append(bounding_box)

    pub_msg.header = header
    pub_msg.bounding_boxes = bounding_boxes

    publisher.publish(pub_msg)

    print face_bbs


openface = OpenfaceROS()
openface.createDlibAlign()

publisher = None

if __name__ == '__main__':
    rospy.init_node('face_detection_node', anonymous = True)

    rospy.Subscriber('/cv_camera/image_raw', Image, imageListener)

    publisher = rospy.Publisher('/vision_system/fr/face_detection', DetectedFaces, queue_size = 100)

    rospy.spin()