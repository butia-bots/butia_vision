#!/usr/bin/env python

import rospy
import argparse

from openface_ros import OpenfaceROS
from vision_system_msgs.msg import ClassifierReload
from vision_system_msgs.srv import FaceClassifierTraining

def classifierTraining(request):
    print('Training ' + request.classifier_name + ' of ' + request.classifier_type + ' type.')
    sucess = openface.trainingProcess(request)
    classifier_reload.publish(ClassifierReload(request.classifier_name))
    print('Trained.')
    ##return FaceClassifierTrainingResponse(sucess)

openface = OpenfaceROS()
classifier_reload = None

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
            'gnb',
            'dbn'],
        help='The type of classifier to use.',
        default='lsvm')


    rospy.init_node('classifier_training_node', anonymous = True)

    training_server = rospy.Service('/vision_system/fr/classifier_training', FaceClassifierTraining, classifierTraining)

    classifier_reload = rospy.Publisher('/vision_system/fr/classifier_reload', ClassifierReload, queue_size = 100)

    args = parser.parse_args()
    if(args.train):
        ros_srv = FaceClassifierTraining()
        ros_srv.classifier_type = args.classifierType
        ros_srv.classifier_name = args.classifierName
        classifierTraining(ros_srv)

    rospy.spin()