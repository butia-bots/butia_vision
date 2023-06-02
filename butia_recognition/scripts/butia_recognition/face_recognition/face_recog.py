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

import pickle

from std_msgs.msg import Header
from sensor_msgs.msg import Image
from butia_vision_msgs.msg import Description2D, Recognitions2D
from butia_vision_msgs.srv import PeopleIntroducing, PeopleIntroducingResponse

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
        for i  in range(0, sec):
            print(str(sec-i) + '...')
            time.sleep(1)
    
    def saveVar(self, variable, filename):
        with open(self.features_dir + '/' +  filename + '.pkl', 'wb') as file:
            pickle.dump(variable, file)

    def loadVar(self, filename):
        file_path = self.features_dir + '/' +  filename + '.pkl'
        if os.path.exists(file_path):
            with open(file_path, 'rb') as file:
                variable = pickle.load(file)
                #print(variable)
            return variable
        return {}

    def flatten(self, l):
        values_list = [item for sublist in l.values() for item in sublist]
        keys_list = [item for name in l.keys() for item in [name]*len(l[name])]
        #print(keys_list)
        return keys_list, values_list


    def encode_faces(self):

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

                # Loop through each training image for the current person
                for person_img in pix:
                    # Get the face encodings for the face in each image file
                    face = face_recognition.load_image_file(self.dataset_dir + person + "/" + person_img)
                    face_bounding_boxes = face_recognition.face_locations(face, model = 'cnn')
                    #print(person, person_img, face_bounding_boxes)

                    #If training image contains exactly one face
                    if len(face_bounding_boxes) > 0:
                        face_enc = face_recognition.face_encodings(face, known_face_locations= face_bounding_boxes)[0]
                        # Add face encoding for current image with corresponding label (name) to the training data
                        encodings.append(face_enc)
                        #print (encodings)
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
    
    #def moreProxFace(self):
        #faceMP = face_recognition.face_locations()
        #facesMP = 
        #if faceMP != None and len(facesMP) > 1:
            #for self.face in faces:

                

    def PeopleIntroducing(self, ros_srv):

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
        while i<num_images:
            self.regressiveCounter(ros_srv.interval)
            #print("vo entra em")
            try:
                #print("Entrei")

                ros_image_aux = rospy.wait_for_message(self.subscribers_dict['image_rgb'], Image, 1000)
            except (ROSException, ROSInterruptException) as e:
                #print(e)
                #print("pq eu to aq")
                break
            ros_image = ros_numpy.numpify(ros_image_aux)
            ros_image = np.flip(ros_image)
            ros_image = np.flipud(ros_image)
            face = face_recognition.face_locations(ros_image, model='cnn')
            image_idx = 0
            if len(face) > 1:
                for idx, (top, right, bottom, left) in enumerate(face):
                    area = (right-left) * (top-bottom)
                    if area > prevArea:
                        prevArea = area
                        image_idx = idx
                
            if len(face) > 0:
                #("detectei o rosto")
                biggest_face = face[image_idx] 
                #if face != None:
                #   if len(face):
                #      top, right, bottom, left = face[0]
                #     cv2.rectangle(s_rgb_image, (left, top), (right, bottom), (0, 0, 255), 2)

                #cv2.imshow("Person", s_rgb_image)
                #print('TEM FACE')
                cv2.imwrite(os.path.join(NAME_DIR, add_image_labels[i]), ros_image)
                rospy.logwarn('Picture ' + add_image_labels[i] + ' was  saved.')
                i+= 1
            else:
                rospy.logerr("The face was not detected.")


        cv2.destroyAllWindows()
        response = PeopleIntroducingResponse()
        response.response = True
        self.encode_faces()

        known_faces_dict = self.loadVar('features')
        self.know_faces = self.flatten(known_faces_dict)
        return response

    @ifState
    def callback(self, *args):
        thold = 0.6
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
        
        rospy.loginfo('Image ID: ' + str(img.header.seq))

        ros_img_small_frame = ros_numpy.numpify(img)

        current_faces = face_recognition.face_locations(ros_img_small_frame, model = 'yolov8')
        current_faces_encodings = face_recognition.face_encodings(ros_img_small_frame, current_faces)

        
        
        for current_encoding in current_faces_encodings:
            description = Description2D()
            name = 'unknown'
            if(len(self.know_faces[0])>0):
                face_distances = np.linalg.norm(self.know_faces[1] - current_encoding, axis = 1)
                min_distance_idx = np.argmin(face_distances)
                min_distance = face_distances[min_distance_idx]
                if min_distance < thold:
                    name = (self.know_faces[0][min_distance_idx])

            description.label = name

            face_rec.descriptions.append(description)

        debug_img = copy(ros_img_small_frame)

        # Display the results
        for idx, (top, right, bottom, left) in enumerate(current_faces):
            print(top, right, bottom, left)
            description_header = img.header
            description_header.seq = 0
            description.header = copy(description_header)
            description.type = Description2D.DETECTION
            description.id = description.header.seq
            description.score = 1
            size = int(right-left), int(bottom-top)
            description.bbox.center.x = int(top) + int(size[0]/2)
            description.bbox.center.y = int(left) + int(size[1]/2)
            description.bbox.size_x = right-left
            description.bbox.size_y = bottom-top

            cv2.rectangle(debug_img, (left, top), (right, bottom), (0, 255, 0), 2)

            # Draw a label with a name below the faceface_recognition_node
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(debug_img, str(face_rec.descriptions[idx].label), (left + 4, bottom - 4), font, 0.5, (0,0,255), 2)
            description_header.seq += 1
        
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
        rospy.loginfo('foi 1')


if __name__ == '__main__':
    rospy.init_node('face_recognition_node', anonymous = True)
    
    face_rec = FaceRecognition()

    rospy.spin()
