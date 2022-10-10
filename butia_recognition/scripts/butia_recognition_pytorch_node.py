#!/usr/bin/env python

import rospy

from butia_recognition import YoloV5Recognition

if __name__ == '__main__':
    rospy.init_node('butia_recognition_pytorch_node', anonymous = True)

    yolo = YoloV5Recognition()

    rospy.spin()
