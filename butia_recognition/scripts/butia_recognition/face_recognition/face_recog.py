#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import rospy

import ros_numpy

from butia_recognition import BaseRecognition, ifState

import numpy as np
import os
from copy import copy
import cv2
import time
import rospkg
from deepface import DeepFace
from std_msgs.msg import Header
from sensor_msgs.msg import Image
from butia_vision_msgs.msg import Description2D, Recognitions2D
from butia_vision_msgs.srv import PeopleIntroducing, PeopleIntroducingResponse
from geometry_msgs.msg import Vector3
from cv_bridge import CvBridge
import os
import sys

bridge = CvBridge()

PACK_DIR = rospkg.RosPack().get_path('butia_recognition')

class SuppressOutput:
    def __enter__(self):
        self._stdout = sys.stdout
        self._stderr = sys.stderr
        sys.stdout = open(os.devnull, 'w')
        sys.stderr = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stderr.close()
        sys.stdout = self._stdout
        sys.stderr = self._stderr
 
class FaceRecognition(BaseRecognition):
    def __init__(self, state=True):
        super().__init__(state=state)
        self.dataset_dir = os.path.join(PACK_DIR, 'dataset')
        os.makedirs(self.dataset_dir, exist_ok=True)

        # Create a test directory inside the dataset directory with a single black image
        test_dir = os.path.join(self.dataset_dir, 'test')
        os.makedirs(test_dir, exist_ok=True)
        cv2.imwrite(os.path.join(test_dir, 'black.jpg'), np.zeros((1, 1, 3), dtype=np.uint8))
        rospy.loginfo(f'Test directory created at {test_dir} with a black image.')
        
        self.readParameters()
        self.initRosComm()

    def initRosComm(self):
        self.debug_publisher = rospy.Publisher(self.debug_topic, Image, queue_size=self.debug_qs)
        self.face_recognition_publisher = rospy.Publisher(self.face_recognition_topic, Recognitions2D, queue_size=self.face_recognition_qs)
        self.introduct_person_service = rospy.Service(self.introduct_person_servername, PeopleIntroducing, self.PeopleIntroducing)

        super().initRosComm(callbacks_obj=self)
        rospy.loginfo('foi 2')

    def regressiveCounter(self, sec):
        sec = int(sec)
        for i  in range(0, sec):
            print(str(sec-i) + '...')
            time.sleep(1)

    def PeopleIntroducing(self, ros_srv):
        name = ros_srv.name
        num_images = ros_srv.num_images

        NAME_DIR = os.path.join(self.dataset_dir, name)
        os.makedirs(NAME_DIR, exist_ok=True)

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
        while i < num_images:

            self.regressiveCounter(ros_srv.interval)

            faceMessage = rospy.wait_for_message('/butia_vision/br/face_recognition', Recognitions2D, 1000)
            image = faceMessage.image_rgb
            ros_image = ros_numpy.numpify(image)

            croped_face_image = None
            ros_image_cv = cv2.cvtColor(ros_image, cv2.COLOR_RGB2BGR)
            for faceInfos in faceMessage.descriptions:
                    if faceInfos.label == 'unknown' or faceInfos.label == name:
                        bbox = faceInfos.bbox
                        biggest_area = 0
                        detected_area = bbox.size_x * bbox.size_y
                        if detected_area > biggest_area:
                            biggest_area = detected_area

                            left = int(bbox.center.x - bbox.size_x / 2)
                            right = int(bbox.center.x + bbox.size_x / 2)
                            top = int(bbox.center.y + bbox.size_y / 2)
                            bottom = int(bbox.center.y - bbox.size_y / 2)

                            croped_face_image = ros_image_cv[
                                bottom:top,
                                left:right
                            ]
                            
            if croped_face_image is not None:
                cv2.imwrite(os.path.join(NAME_DIR, add_image_labels[i]), croped_face_image)
                rospy.logwarn('Picture ' + add_image_labels[i] + ' was  saved.')
                i+= 1
            else:
                rospy.logerr("The face was not detected.")

        cv2.destroyAllWindows()
        response = PeopleIntroducingResponse()
        response.response = True

        return response
    
    @ifState
    def callback(self, *args):
        source_data = self.sourceDataFromArgs(args)

        if 'image_rgb' not in source_data:
            rospy.logwarn('Souce data has no image_rgb.')
            return None
        
        img = source_data['image_rgb']

        thold = 0.5
        names = []
        h = Header()
        h.seq = self.seq
        self.seq += 1
        h.stamp = rospy.Time.now()
        face_rec = Recognitions2D()
        face_rec.header = h
        ros_img_small_frame = ros_numpy.numpify(img)

        with SuppressOutput():
            current_faces = DeepFace.extract_faces(
                ros_img_small_frame,
                detector_backend='yolov8',
                enforce_detection=False,
                align=False
            )

        debug_img = copy(ros_img_small_frame)
        for idx in range(len(current_faces)):

            result = current_faces[idx]
            face = (result['face']* 255).astype('uint8')

            x_top = result['facial_area']['x']
            y_top = result['facial_area']['y']
            w = result['facial_area']['w']
            h = result['facial_area']['h']

            x_center = x_top + int(w/2)
            y_center = y_top + int(h/2)

            left = x_center - int(w/2)
            right = x_center + int(w/2)
            top = y_center + int(h/2)
            bottom = y_center - int(h/2)

            description = Description2D()
            name = 'unknown'

            #If there are known faces, compare the current face with them
            with SuppressOutput():
                detected_face_result = DeepFace.find(
                    img_path=face,
                    db_path=self.dataset_dir,
                    model_name="Dlib",  
                    detector_backend = 'skip',
                    enforce_detection=False,
                    threshold=thold
                )

            if detected_face_result[0].empty:
                pass
            else:
                final_image_path = detected_face_result[0].iloc[0]['identity']
                name = final_image_path.split("/")[-2]

            description.label = name
            names.append(name)

            description_header = img.header
            description_header.seq = 0
            description.header = copy(description_header)
            description.type = Description2D.DETECTION
            description.id = description.header.seq
            description.score = 1
            description.max_size = Vector3(*[0.2, 0.2, 0.2])
            description.bbox.center.x = x_center
            description.bbox.center.y = y_center
            description.bbox.size_x = w
            description.bbox.size_y = h

            cv2.rectangle(debug_img, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(debug_img, name, (left + 4, bottom - 4), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0,0,255), 2)
            description_header.seq += 1
            face_rec.descriptions.append(description)
            face_rec.image_rgb = img
            
        self.debug_publisher.publish(ros_numpy.msgify(Image, debug_img, 'rgb8'))
        if len(face_rec.descriptions) > 0:
            self.face_recognition_publisher.publish(face_rec)

    def readParameters(self):
        self.debug_topic = rospy.get_param("~publishers/debug/topic", "/butia_vision/br/debug")
        self.debug_qs = rospy.get_param("~publishers/debug/queue_size", 1)

        self.face_recognition_topic = rospy.get_param("~publishers/face_recognition/topic", "/butia_vision/br/face_recognition")

        self.face_recognition_qs = rospy.get_param("~publishers/face_recognition/queue_size", 1)

        self.introduct_person_servername = rospy.get_param("~servers/introduct_person/servername", "/butia_vision/br/introduct_person")

        super().readParameters()


if __name__ == '__main__':
    rospy.init_node('face_recognition_node', anonymous = True)
    
    face_rec = FaceRecognition()

    rospy.spin()
