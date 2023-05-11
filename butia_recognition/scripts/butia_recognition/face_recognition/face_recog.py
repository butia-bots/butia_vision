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

from std_msgs.msg import Header
from sensor_msgs.msg import Image
from butia_vision_msgs.msg import Description2D, Recognitions2D

torch.set_num_threads(1)

class FaceRecognition(BaseRecognition):
    def __init__(self, state=True):
        super().__init__(state=state)
        self.readParameters()

        self.initRosComm()

        self.face_enconder()

    def initRosComm(self):
        self.debug_publisher = rospy.Publisher(self.debug_topic, Image, queue_size=self.debug_qs)
        self.face_recognition_publisher = rospy.Publisher(self.face_recognition_topic, Recognitions2D, queue_size=self.face_recognition_qs)
        super().initRosComm(callbacks_obj=self)
        rospy.loginfo('foi 2')

    def face_enconder(self):

        image_path = '/home/andremaurell/butia_ws/src/butia_vision/butia_recognition/include/face_rec_images/'
        
        # Load a sample picture and learn how to recognize it.
        obama_image = face_recognition.load_image_file(os.path.join(image_path, "barackObama.jpg"))
        obama_face_encoding = face_recognition.face_encodings(obama_image)[0]

        # Load a second sample picture and learn how to recognize it.
        elon_image = face_recognition.load_image_file(os.path.join(image_path, "elonMusk.jpg"))
        elon_face_encoding = face_recognition.face_encodings(elon_image)[0]

        woods_image = face_recognition.load_image_file(os.path.join(image_path, "tigerWoods.jpeg"))
        woods_face_encoding = face_recognition.face_encodings(woods_image)[0]

        nico_image = face_recognition.load_image_file(os.path.join(image_path, "nicolainielsen.png"))
        nico_face_encoding = face_recognition.face_encodings(nico_image)[0]

        # Create arrays of known face encodings and their names
        self.known_face_encodings = [
            obama_face_encoding,
            elon_face_encoding,
            woods_face_encoding,
            nico_face_encoding
        ]
        self.known_face_names = [
            "Barack Obama",
            "Elon Musk",
            "Tiger Woods",
            "Nicolai Nielsen"
        ]
        rospy.loginfo('foi 3')
    @ifState
    def callback(self, *args):
        print("callback")
        # Initialize some variables
        face_locations = []
        face_encodings = []
        face_names = []
        process_this_frame = True

        face_rec = Recognitions2D()
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
            print("Process??")
            # Find all the faces and face encodings in the current frame of video
            face_locations = face_recognition.face_locations(cv_img_small_frame, model = 'yolov8')
            face_encodings = face_recognition.face_encodings(cv_img_small_frame, face_locations)
            #print(face_locations)
            #print(face_encodings)
            face_names = []
            for face_encoding in face_encodings:
                print("entra no for 1")
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
                    print("ve os matchs")
                    name = self.known_face_names[best_match_index]
                    description.label = name
                face_rec.descriptions.append(Description2D)

        process_this_frame = not process_this_frame

        
        debug_img = copy(cv_img_small_frame)

        h = Header()
        h.seq = self.seq
        self.seq += 1
        h.stamp = rospy.Time.now()
        face_rec.header = h
        face_rec.image_rgb = copy(img)
        face_rec.points = copy(points)

        description_header = img.header
        description_header.seq = 0
        #print(face_locations)
        #print(face_names)
        # Display the results
        for (top, right, bottom, left), name in zip(face_locations, face_encodings):
            print("reescala")
            # Scale back up face locations since the frame we detected in was scaled to 1/4 size
            #top *= 4
            #right *= 4
            #bottom *= 4
            #left *= 4
              
            description = Description2D()
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
            print("existe o description")
            self.face_recognition_publisher.publish(face_rec)

    def readParameters(self):
        self.debug_topic = rospy.get_param("~publishers/debug/topic", "/butia_vision/br/debug")
        self.debug_qs = rospy.get_param("~publishers/debug/queue_size", 1)

        self.face_recognition_topic = rospy.get_param("~publishers/face_recognition/topic", "/butia_vision/br/face_recognition")
        self.face_recognition_qs = rospy.get_param("~publishers/face_recognition/queue_size", 1)

        super().readParameters()
        rospy.loginfo('foi 1')

if __name__ == '__main__':
    rospy.init_node('face_recognition_node', anonymous = True)
    
    face_rec = FaceRecognition()

    rospy.spin()
