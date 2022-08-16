#!/usr/bin/env python

import rospy

from vision_bridge import VisionBridge

if __name__ == '__main__':
    rospy.init_node('butia_vision_bridge_node', anonymous = True)

    vision_bridge = VisionBridge()

    rospy.spin()
