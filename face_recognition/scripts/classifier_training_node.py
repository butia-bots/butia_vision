#!/usr/bin/env python

import rospy

from openface_ros import OpenfaceROS
from vision_system_msgs.srv import FaceClassifierTraining

def classifierTraining(request):
    openface.trainClassifier(request.classifier_type)

openface = OpenfaceROS()

if __name__ == '__main__':
    rospy.init_node('classifier_training_node', anonymous = True)

    training_server = rospy.Service('/vision_system/fr/classifier_training', FaceClassifierTraining,classifierTraining, )

    rospy.spin()