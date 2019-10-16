#!/usr/bin/env python

import rospy

from yolo_recognition import YoloRecognition

if __name__ == '__main__':
    rospy.init_node('object_recogniton_node', anonymous = True)

    yolo = YoloRecognition()

    rospy.spin()
