#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import rospy

import ros_numpy

from butia_recognition import BaseRecognition, ifState

import torch
import numpy as np
import os
from copy import copy
import cv2
import face_recognition
import time
from openface import helper

from std_msgs.msg import Header
from sensor_msgs.msg import Image
from butia_vision_msgs.msg import Description2D, Recognitions2D
from butia_vision_msgs.srv import PeopleIntroducing
class FaceRecognition(BaseRecognition):
    def __init__(self, state=True):
        super().__init__(state=state)
        self.readParameters()

        self.initRosComm()

        self.face_enconder()

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

    def encode_faces(self):

        train_direct = '/home/butiabots/Desktop/testes_joao/face_recognition/train_dir/'
        encodings = []
        names = []
        encoded_face = {}

        train_dir = os.listdir(train_direct)

        for person in train_dir:
            pix = os.listdir(train_direct + person)

            # Loop through each training image for the current person
            for person_img in pix:
                # Get the face encodings for the face in each image file
                face = face_recognition.load_image_file(train_direct + person + "/" + person_img)
                print(person)
                face_bounding_boxes = face_recognition.face_locations(face)

                #If training image contains exactly one face
                if len(face_bounding_boxes) == 1:
                    face_enc = face_recognition.face_encodings(face)[0]
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
        print(encoded_face)
    

    def PeopleIntroducing(self, ros_srv):

        DATASET_DIR = '/home/butiabots/Desktop/testes_joao/face_recognition/train_dir/'

        name = ros_srv.name
        num_images = ros_srv.num_images
        NAME_DIR = os.path.join(DATASET_DIR, name)
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
            self.regressiveCounter(ros_srv.interval)
            print("vo entra em")
            try:
                print("Entrei")

                ros_image_aux = rospy.wait_for_message(self.image_topic, Image, 1000)
            except (ROSException, ROSInterruptException) as e:
                print(e)
                print("pq eu to aq")
                break
            ros_image = ros_numpy.numpify(ros_image_aux)
            ros_image = np.flip(ros_image)
            ros_image = np.flipud(ros_image)
            face = face_recognition.face_locations(ros_image, model='cnn')
            print("detectei o rosto")
            s_rgb_image = ros_image.copy() 
            #if face != None:
             #   if len(face):
              #      top, right, bottom, left = face[0]
               #     cv2.rectangle(s_rgb_image, (left, top), (right, bottom), (0, 0, 255), 2)

            cv2.imshow("Person", s_rgb_image)

            if face != None:
                rospy.logwarn('Picture ' + add_image_labels[i] + ' was  saved.')
                cv2.imwrite(os.path.join(NAME_DIR, add_image_labels[i]), ros_image)
                i+= 1
            else:
                rospy.logerr("The face was not detected.")


        cv2.destroyAllWindows()

        return self.encode_faces()
    
    @ifState
    def callback(self, *args):

        # Initialize some variables
        # face_names = []
        face_locations = []
        face_encodings = []
        process_this_frame = True

        face_rec = Recognitions2D()
        description = Description2D()
        img = None
        points = None
        if len(args):
            img = args[0]
            points = args[1]

        rospy.loginfo('Image ID: ' + str(img.header.seq))

        #cv_img = np.flip(ros_numpy.numpify(img),2)
        cv_img = ros_numpy.numpify(img)

        # Resize frame of video to 1/4 size for faster face recognition processing
        #small_frame = cv2.resize(cv_img, (0, 0), fx=0.25, fy=0.25)

        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        #rgb_small_frame = small_frame[:, :, ::-1]
        cv_img_small_frame = cv_img[:, :, ::-1]


        # Only process every other frame of video to save time
        if process_this_frame:

            # Find all the faces and face encodings in the current frame of video
            face_locations = face_recognition.face_locations(cv_img_small_frame, model = 'cnn')
            face_encodings = face_recognition.face_encodings(cv_img_small_frame, face_locations)
            #print(face_locations)
            #print(face_encodings)
            face_names = []
            for face_encoding in face_encodings:

                # See if the face is a match for the known face(s)
                matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
                name = "Unknown"

                # # If a match was found in known_face_encodings, just use the first one.
                # if True in matches:
                #     first_match_index = matches.index(True)
                #     name = known_face_names[first_match_index]

                # Or instead, use the known face with the smallest distance to the new face
                face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)

                if matches[best_match_index]:
                    name = self.known_face_names[best_match_index]

                description.label = name

                face_rec.descriptions.append(description)

        process_this_frame = not process_this_frame

        
        debug_img = copy(cv_img_small_frame)

        h = Header()
        h.seq = self.seq
        self.seq += 1
        h.stamp = rospy.Time.now()


        #print(face_locations) self.image_topic
        # Display the results
        for (top, right, bottom, left), name in zip(face_locations, face_encodings):

            # Scale back up face locations since the frame we detected in was scaled to 1/4 size
            #top *= 4
            #right *= 4
            #bottom *= 4
            #left *= 4
            description_header = img.header
            description_header.seq = 0
            description.header = copy(description_header)
            description.type = Description2D.DETECTION
            description.id = description.header.seq
            description.score = 1
            size = int(right-left), int(top-bottom)
            description.bbox.center.x = int(bottom) + int(size[0]/2)
            description.bbox.center.y = int(left) + int(size[1]/2)
            description.bbox.size_x = right-left
            description.bbox.size_y = top-bottom

            cv2.rectangle(debug_img, (left, top), (right, bottom), (0, 255, 0), 2)

            # Draw a label with a name below the faceface_recognition_node
            font = cv2.FONT_HERSHEY_DUPLEX
            #cv2.putText(debug_img, str(name), (left + 4, bottom - 4), font, 1.0, (0,0,255), 2)
            description_header.seq += 1
        
        self.debug_publisher.publish(ros_numpy.msgify(Image, debug_img, 'rgb8'))

        if len(face_rec.descriptions) > 0:
            print('vou publicar ein')
            self.face_recognition_publisher.publish(face_rec)

    def readParameters(self):
        self.debug_topic = rospy.get_param("~publishers/debug/topic", "/butia_vision/br/debug")
        self.debug_qs = rospy.get_param("~publishers/debug/queue_size", 1)

        self.face_recognition_topic = rospy.get_param("~publishers/face_recognition/topic", "/butia_vision/br/face_recognition")
        self.face_recognition_qs = rospy.get_param("~publishers/face_recognition/queue_size", 1)

        self.image_topic = rospy.get_param('/face_recognition/subscribers/camera_reading/topic', '/usb_cam/image_raw')

        self.introduct_person_servername = rospy.get_param("~servers/introduct_person/servername", "/butia_vision/br/introduct_person")

        super().readParameters()
        rospy.loginfo('foi 1')


if __name__ == '__main__':
    rospy.init_node('face_recognition_node', anonymous = True)
    
    face_rec = FaceRecognition()

    rospy.spin()
