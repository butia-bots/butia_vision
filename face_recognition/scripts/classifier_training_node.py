#!/usr/bin/env python

import rospy

from face_recognition_ros import FaceRecognitionROS
from vision_system_msgs.msg import ClassifierReload
from vision_system_msgs.srv import FaceClassifierTraining, FaceClassifierTrainingResponse

def classifierTraining(ros_srv):
        ans = face_recognition_ros.trainingProcess(ros_srv)
        #if(ans):
        #    classifier_reload.publish(ClassifierReload(ros_srv.classifier_name))
        #return ans

face_recognition_ros = FaceRecognitionROS()
classifier_reload = None
training_server = None

if __name__ == '__main__':
    classifier_reload_topic = rospy.get_param('/face_recognition/publishers/classifier_reload/topic', '/vision_system/fr/classifier_reload')
    classifier_reload_qs = rospy.get_param('/face_recognition/publishers/classifier_reload/queue_size', 1)

    classifier_training_service = rospy.get_param('/face_recognition/services/classifier_training/service','/vision_system/fr/classifier_training')

    rospy.init_node('classifier_training_node', anonymous = True)

    training_server = rospy.Service(classifier_training_service, FaceClassifierTraining, classifierTraining)

    classifier_reload = rospy.Publisher(classifier_reload_topic, ClassifierReload, queue_size = classifier_reload_qs)

    rospy.spin()