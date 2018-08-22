#!/usr/bin/env python

import rospy

from openface_ros import OpenfaceROS
from vision_system_msgs.srv import PeopleIntroducing

def peopleIntroducing(request):
    rc = openface.introducingProcess(request)


openface = OpenfaceROS()

if __name__ == '__main__':
    introducing_server = rospy.Service('/vision_system/fr/people_introducing', PeopleIntroducing, peopleIntroducing)