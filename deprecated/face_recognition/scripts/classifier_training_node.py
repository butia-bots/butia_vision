#!/usr/bin/env python

import rospy
import rospkg
import time

from face_recognition_ros import FaceRecognitionROS
from butia_vision_msgs.msg import ClassifierReload
from butia_vision_msgs.srv import FaceClassifierTraining, FaceClassifierTrainingResponse, PeopleIntroducing

from sensor_msgs.msg import Image
import cv2
from cv_bridge import CvBridge
import os
from openface import helper
import numpy as np

BRIDGE = CvBridge()
DATASET_DIR = os.path.join(rospkg.RosPack().get_path('face_recognition'), 'dataset')

def classifierTraining(ros_srv):
    ans = face_recognition_ros.trainingProcess(ros_srv)
    if(ans):
        classifier_reload.publish(ClassifierReload(ros_srv.classifier_name))
    return ans

def regressiveCounter(sec):
    for i  in range(0, sec):
        print(str(sec-i) + '...')
        time.sleep(1)
        

def peopleIntroducing(ros_srv):
    name = ros_srv.name
    num_images = ros_srv.num_images
    NAME_DIR = os.path.join(DATASET_DIR, 'raw', name)
    helper.mkdirP(NAME_DIR)

    image_type = '.jpg'

    image_labels = os.listdir(NAME_DIR)
    add_image_labels = []
    i = 1
    k = 0
    j = num_images
    number = [] 
    for label in image_labels:
        number.append(int(label.replace(image_type, '')))
    
    number.sort()
    n = 1
    while j > 0:
        if k < len(number):
            n = number[k] + 1
            if number[k] == i:
                k += 1
            else:
                add_image_labels.append((str(i) + image_type))
                j -= 1      
            i += 1 

        else:
            add_image_labels.append(str(n) + image_type)
            j -= 1
            n += 1
    
    num_images = ros_srv.num_images

    i = 0
    while i<num_images:
        try:
            ros_image = rospy.wait_for_message(image_topic, Image, 1000)
        except (ROSException, ROSInterruptException) as e:
            print(e)
            break

        rgb_image = BRIDGE.imgmsg_to_cv2(ros_image, desired_encoding="bgr8")

        face_recognition_ros.image_width = ros_image.width
        face_recognition_ros.image_height = ros_image.height

        face = face_recognition_ros.detectLargestFace(rgb_image)
        s_rgb_image = rgb_image.copy() 
        if face != None:
            bb = face_recognition_ros.dlibRectangle2RosBoundingBox(face)
            color = (0, 255, 0)
            cv2.rectangle(s_rgb_image, (bb.minX, bb.minY), (bb.minX + bb.width, bb.minY + bb.height), color, 2)    

        cv2.imshow("Person", s_rgb_image)
        
        regressiveCounter(3)

        if face != None:
            rospy.logwarn('Picture ' + add_image_labels[i] + ' was  saved.')
            cv2.imwrite(os.path.join(NAME_DIR, add_image_labels[i]), rgb_image)
            i+= 1
        else:
            rospy.logerr("The face was not detected.")


    cv2.destroyAllWindows()

    classifier_training = FaceClassifierTraining()
    classifier_training.classifier_type = ros_srv.classifier_type
    classifier_training.classifier_name = 'classifier_' + ros_srv.classifier_type + '_' + name + '.pkl'

    return classifierTraining(classifier_training)


face_recognition_ros = FaceRecognitionROS()
image_topic = None
classifier_reload = None
training_server = None

if __name__ == '__main__':
    classifier_reload_topic = rospy.get_param('/face_recognition/publishers/classifier_reload/topic', '/butia_vision/fr/classifier_reload')
    classifier_reload_qs = rospy.get_param('/face_recognition/publishers/classifier_reload/queue_size', 1)

    image_topic = rospy.get_param('/face_recognition/subscribers/camera_reading/topic', '/butia_vision/bvb/image_rgb_raw')
    people_introducing_service = rospy.get_param('/face_recognition/servers/people_introducing/service', '/butia_vision/fr/people_introducing')
    classifier_training_service = rospy.get_param('/face_recognition/servers/classifier_training/service','/butia_vision/fr/classifier_training')

    rospy.init_node('classifier_training_node', anonymous = True)

    training_server = rospy.Service(classifier_training_service, FaceClassifierTraining, classifierTraining)
    people_introducing_server = rospy.Service(people_introducing_service, PeopleIntroducing, peopleIntroducing)

    classifier_reload = rospy.Publisher(classifier_reload_topic, ClassifierReload, queue_size = classifier_reload_qs)

    rospy.spin()
