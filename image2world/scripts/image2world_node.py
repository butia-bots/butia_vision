#!/usr/bin/env python
import rospy
from vision_system_msgs import Recognitions

def image2worldCallback(message):
    image_id = message.image_header.seq

    rgb_image = #service
    depth_image = #service

    #segmentation service

    point_cloud = []