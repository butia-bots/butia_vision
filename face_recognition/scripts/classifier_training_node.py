#!/usr/bin/env python

import rospy
import argparse

from openface_ros import OpenfaceROS
from vision_system_msgs.msg import ClassifierReload
from vision_system_msgs.srv import FaceClassifierTraining, FaceClassifierTrainingResponse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true')
    parser.add_argument(
        '--classifierName',
        type=str,
        help='Name of classifier to be stored .pkl.',
        default='classifier.pkl')
    parser.add_argument(
        '--classifierType',
        type=str,
        choices=[
            'lsvm',
            'gssvm',
            'gmm',
            'rsvm',
            'dt',
            'gnb'],
        help='The type of classifier to use.',
        default='lsvm')

    classifier_reload_topic = rospy.get_param('/face_recognition/publishers/classifier_reload/topic', '/vision_system/fr/classifier_reload')
    classifier_reload_qs = rospy.get_param('/face_recognition/publishers/classifier_reload/queue_size', 1)

    classifier_training_service = rospy.get_param('/face_recognition/services/classifier_training/service','/vision_system/fr/classifier_training')

    rospy.init_node('classifier_training_node', anonymous = True)

    training_server = rospy.Service(classifier_training_service, FaceClassifierTraining, classifierTraining)

    classifier_reload = rospy.Publisher(classifier_reload_topic, ClassifierReload, classifier_reload_qs)

    '''args = parser.parse_args()
    if(args.train):
        ros_srv = FaceClassifierTraining()
        ros_srv.classifier_type = args.classifierType
        ros_srv.classifier_name = args.classifierName
        classifierTraining(ros_srv) '''

    rospy.spin()