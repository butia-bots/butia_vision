#!/usr/bin/env python

import rospy

from yolo_recognition import YoloRecognition

if __name__ == '__main__':
    rospy.init_node('butia_recognition_pytorch_node', anonymous = True)

    yolo = YoloRecognition()

    rospy.spin()
