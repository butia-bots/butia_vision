#!/usr/bin/env python

import rospy

from butia_recognition import YoloV8Recognition

if __name__ == '__main__':
    rospy.init_node('butia_recognition_pytorch_node', anonymous = True)

    yolo = YoloV8Recognition()

    rospy.spin()
