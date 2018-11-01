#!/usr/bin/env python

import rospy
import rospkg

from face_recognition_ros import FaceRecognitionROS
from vision_system_msgs.msg import ClassifierReload
from vision_system_msgs.srv import FaceClassifierTraining, FaceClassifierTrainingResponse, PeopleIntroducing

from sensor_msgs.msg import Image
import cv2
from cv_bridge import CvBridge

BRIDGE = CvBridge()
PACK_DIR = rospkg.RosPack().get_path('face_recognition')

def classifierTraining(ros_srv):
    ans = face_recognition_ros.trainingProcess(ros_srv)
    if(ans):
        classifier_reload.publish(ClassifierReload(ros_srv.classifier_name))
    return ans

def peopleIntroducing(ros_srv):
    name = ros_srv.req.name
    num_images = ros_srv.req.num_images

    i = 0
    while i<num_images:
        ros_image = rospy.wait_for_message(image_topic, Image, 1000)

        rgb_image = BRIDGE.imgmsg_to_cv2(ros_msg, desired_encoding="rgb8")

        cv2.imshow("Person", rgb_image)

        cv2.waitKey(1) == 32:
            i+= 1


face_recognition_ros = FaceRecognitionROS()
image_topic = None
classifier_reload = None
training_server = None

if __name__ == '__main__':
    classifier_reload_topic = rospy.get_param('/face_recognition/publishers/classifier_reload/topic', '/vision_system/fr/classifier_reload')
    classifier_reload_qs = rospy.get_param('/face_recognition/publishers/classifier_reload/queue_size', 1)

    image_topic = rospy.get_param('/face_recognition/subscribers/camera_reading/topic', '/vision_system/vsb/image_rgb_raw')
    people_introducing_service = rospy.get_param('/face_recognition/services/people_introducing/service', '/vision_system/fr/people_introducing')
    classifier_training_service = rospy.get_param('/face_recognition/services/classifier_training/service','/vision_system/fr/classifier_training')

    rospy.init_node('classifier_training_node', anonymous = True)

    training_server = rospy.Service(classifier_training_service, FaceClassifierTraining, classifierTraining)
    people_introducing_server = rospy.Service(people_introducing_service, PeopleIntroducing, peopleIntroducing)

    classifier_reload = rospy.Publisher(classifier_reload_topic, ClassifierReload, queue_size = classifier_reload_qs)

    rospy.spin()