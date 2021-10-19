#!/usr/bin/env python

import rospy

from instance_segmentation import MaskRCNNSegmentation

if __name__ == '__main__':
    rospy.init_node('instance_segmentation_node', anonymous = True)

    maskrcnn = MaskRCNNSegmentation()

    rospy.spin()