#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import rospy

import ros_numpy

from butia_recognition import BaseRecognition, ifState

import numpy as np
import os
from copy import copy
import cv2
import face_recognition
import time
import rospkg
from collections import Counter
import pickle

from std_msgs.msg import Header
from sensor_msgs.msg import Image
from butia_vision_msgs.msg import Description2D, Recognitions2D
from butia_vision_msgs.srv import PeopleIntroducing, PeopleIntroducingResponse
from geometry_msgs.msg import Vector3

PACK_DIR = rospkg.RosPack().get_path('butia_recognition')
class FaceRecognition(BaseRecognition):
    def __init__(self, state=True):
        super().__init__(state=state)
        dataset_dir = os.path.join(PACK_DIR, 'dataset')
        self.features_dir = os.path.join(dataset_dir, 'features')
        self.dataset_dir = os.path.join(dataset_dir, 'people/')
        self.readParameters()

        self.initRosComm()

        known_faces_dict = self.loadVar('features')
        self.know_faces = self.flatten(known_faces_dict)

    def initRosComm(self):
        self.debug_publisher = rospy.Publisher(self.debug_topic, Image, queue_size=self.debug_qs)
        self.face_recognition_publisher = rospy.Publisher(self.face_recognition_topic, Recognitions2D, queue_size=self.face_recognition_qs)
        self.introduct_person_service = rospy.Service(self.introduct_person_servername, PeopleIntroducing, self.PeopleIntroducing) #possivelmente trocar self.encode_faces

        super().initRosComm(callbacks_obj=self)
        rospy.loginfo('foi 2')

    def regressiveCounter(self, sec):
        try:
            sec = int(sec)
            for i  in range(0, sec):
                rospy.logwarn(str(sec-i) + '...')
                time.sleep(1)
        except KeyError as e:
            while True:
                rospy.logwarn(f"Regressive counte erro {e}")
    
    def saveVar(self, variable, filename):
        try:
            with open(self.features_dir + '/' +  filename + '.pkl', 'wb') as file:
                pickle.dump(variable, file)
        except KeyError as e:
            while True:
                rospy.logwarn(f"Save var counte erro {e}")

    def loadVar(self, filename):
        try:
            file_path = self.features_dir + '/' +  filename + '.pkl'
            if os.path.exists(file_path):
                with open(file_path, 'rb') as file:
                    variable = pickle.load(file)
                return variable
            return {}
        except KeyError as e:
            while True:
                rospy.logwarn(f"LoadVar counte erro {e}")

    def flatten(self, l):
        try:
            values_list = [item for sublist in l.values() for item in sublist]
            keys_list = [item for name in l.keys() for item in [name]*len(l[name])]
            return keys_list, values_list
        except KeyError as e:
            while True:
                rospy.logwarn(f"flatten counte erro {e}")

    def encode_faces(self, face_bboxes = None):
        try:

            encodings = []
            names = []
            try:
                encoded_face = self.loadVar('features')
            except:
                encoded_face = {}
            train_dir = os.listdir(self.dataset_dir)

            for person in train_dir:
                if person not in self.know_faces[0]:   
                    pix = os.listdir(self.dataset_dir + person)

                    for person_img in pix:
                        if face_bboxes is not None:
                            face_bbox = face_bboxes.pop(0)

                            center_x = face_bbox.center.x
                            center_y = face_bbox.center.y
                            size_x = face_bbox.size_x
                            size_y = face_bbox.size_y

                            left = center_x - size_x / 2
                            right = center_x + size_x / 2
                            top = center_y + size_y / 2
                            bottom = center_y - size_y / 2

                            face_bounding_boxes.append([top, right, bottom, left])
            
                        else:
                            face = face_recognition.load_image_file(self.dataset_dir + person + "/" + person_img)
                            face_bounding_boxes = face_recognition.face_locations(face, model = 'yolov8')

                        M_face = None
                        M_area = -float('inf')
                        for top, right, bottom, left in face_bounding_boxes:
                            area = (bottom - top)*(right - left)
                            if area > M_area:
                                M_area = area
                                M_face = (top, right, bottom, left)

                        if M_face is not None:
                            face_enc = face_recognition.face_encodings(face, known_face_locations=[M_face])[0]
                            encodings.append(face_enc)

                            if person not in names:
                                names.append(person)
                                encoded_face[person] = []
                                encoded_face[person].append(face_enc)
                            else:
                                encoded_face[person].append(face_enc)
                        else:
                            print(person + "/" + person_img + " was skipped and can't be used for training")
                else:
                    pass
            self.saveVar(encoded_face, 'features')             
        except KeyError as e:
            while True:
                rospy.logwarn(f"encode counte erro {e}")
    def PeopleIntroducing(self, ros_srv):
        face_bboxes = []
        name = ros_srv.name
        num_images = ros_srv.num_images
        NAME_DIR = os.path.join(self.dataset_dir, name)
        os.makedirs(NAME_DIR, exist_ok=True)
        os.makedirs(self.features_dir, exist_ok=True)
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
            try:
                recognized_faces = rospy.wait_for_message('/butia_vision/br/face_recognition', Recognitions2D, 1000)
            except (Exception) as e:
                break
            for face in recognized_faces:
                if face.label == 'unknown':
                    ros_image = cv2.cvtColor(face.mask, cv2.COLOR_BGR2RGB)
                    cv2.imwrite(os.path.join(NAME_DIR, add_image_labels[i]), ros_image)
                    rospy.logwarn('Picture ' + add_image_labels[i] + ' was  saved.')
                    i+= 1
                    face_bboxes.append(face.bbox)
                    break
            else:
                rospy.logerr("The face was not detected.")

        cv2.destroyAllWindows()
        response = PeopleIntroducingResponse()
        response.response = True

        self.encode_faces(face_bboxes)

        known_faces_dict = self.loadVar('features')
        self.know_faces = self.flatten(known_faces_dict)
        return response

    @ifState
    def callback(self, *args):
        try:
            thold = 0.5
            face_rec = Recognitions2D()
            source_data = self.sourceDataFromArgs(args)

            if 'image_rgb' not in source_data:
                rospy.logwarn('Souce data has no image_rgb.')
                return None
            
            img = source_data['image_rgb']
            h = Header()
            h.seq = self.seq
            self.seq += 1
            h.stamp = rospy.Time.now()
            face_rec.header = h
            face_rec = BaseRecognition.addSourceData2Recognitions2D(source_data, face_rec)
            #rospy.loginfo('Image ID: ' + str(img.header.seq))

            ros_img_small_frame = ros_numpy.numpify(img)
            rospy.logwarn(ros_img_small_frame.shape)
            current_faces = face_recognition.face_locations(ros_img_small_frame, model = 'yolov8')
            current_faces_encodings = face_recognition.face_encodings(ros_img_small_frame, current_faces)
            debug_img = copy(ros_img_small_frame)
            names = []
            name_distance=[]
            for idx in range(len(current_faces_encodings)):
                current_encoding = current_faces_encodings[idx]
                top, right, bottom, left = current_faces[idx]
                description = Description2D()
                name = 'unknown'
                if(len(self.know_faces[0]) > 0):
                    face_distances = np.linalg.norm(self.know_faces[1] - current_encoding, axis = 1)
                    min_distance_idx = np.argmin(face_distances)
                    min_distance = face_distances[min_distance_idx]
                    if min_distance < thold:
                        name = (self.know_faces[0][min_distance_idx])
                description.label = name

                names.append(name)

                description_header = img.header
                description_header.seq = 0
                description.header = copy(description_header)
                description.type = Description2D.DETECTION
                description.id = description.header.seq
                description.score = 1
                description.max_size = Vector3(*[0.2, 0.2, 0.2])
                size = int(right-left), int(bottom-top)
                description.bbox.center.x = int(left) + int(size[1]/2)
                description.bbox.center.y = int(top) + int(size[0]/2)
                description.bbox.size_x = bottom-top
                description.bbox.size_y = right-left
                description.mask = ros_numpy.numpify(img)
                cv2.rectangle(debug_img, (left, top), (right, bottom), (0, 255, 0), 2)
                
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(debug_img, name, (left + 4, bottom - 4), font, 0.5, (0,0,255), 2)
                description_header.seq += 1

                face_rec.descriptions.append(description)
       
            self.debug_publisher.publish(ros_numpy.msgify(Image, debug_img, 'bgr8'))
            if len(face_rec.descriptions) > 0:
                self.face_recognition_publisher.publish(face_rec)
        except KeyError as e:
            while True:
                rospy.logwarn(f"callback counte erro {e}")

    def readParameters(self):
        self.debug_topic = rospy.get_param("~publishers/debug/topic", "/butia_vision/br/debug")
        self.debug_qs = rospy.get_param("~publishers/debug/queue_size", 1)

        self.face_recognition_topic = rospy.get_param("~publishers/face_recognition/topic", "/butia_vision/br/face_recognition")

        self.face_recognition_qs = rospy.get_param("~publishers/face_recognition/queue_size", 1)

        self.introduct_person_servername = rospy.get_param("~servers/introduct_person/servername", "/butia_vision/br/introduct_person")

        super().readParameters()
        rospy.loginfo('foi 1')


if __name__ == '__main__':
    rospy.init_node('face_recognition_node', anonymous = True)
    
    face_rec = FaceRecognition()

    rospy.spin()
