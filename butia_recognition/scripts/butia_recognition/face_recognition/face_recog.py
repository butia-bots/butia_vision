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
from butia_vision_msgs.msg import Description2D, Recognitions2D, FaceEncoding, FaceDescription
from butia_vision_msgs.srv import PeopleIntroducing, PeopleIntroducingResponse
from geometry_msgs.msg import Vector3

from butia_world_msgs.srv import RedisCacheReaderSrv, RedisCacheWriterSrv
from std_msgs.msg import Empty

PACK_DIR = rospkg.RosPack().get_path('butia_recognition')
class FaceRecognition(BaseRecognition):
    def __init__(self, state=True):
        super().__init__(state=state)
        dataset_dir = os.path.join(PACK_DIR, 'dataset')
        self.features_dir = os.path.join(dataset_dir, 'features')
        self.dataset_dir = os.path.join(dataset_dir, 'people/')
        self.readParameters()

        self.initRosComm()

        self.cache = self.getCache()
        known_faces_dict = self.loadVar('features')
        self.know_faces = self.flatten(known_faces_dict)
        #self.encode_faces()

    def initRosComm(self):
        self.debug_publisher = rospy.Publisher(self.debug_topic, Image, queue_size=self.debug_qs)
        self.face_recognition_publisher = rospy.Publisher(self.face_recognition_topic, Recognitions2D, queue_size=self.face_recognition_qs)
        self.introduct_person_service = rospy.Service(self.introduct_person_servername, PeopleIntroducing, self.PeopleIntroducing) #possivelmente trocar self.encode_faces

        super().initRosComm(callbacks_obj=self)
        rospy.loginfo('foi 2')

    def regressiveCounter(self, sec):
        sec = int(sec)
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
            return variable
        return {}

    def flatten(self, l):
        values_list = [item for sublist in l.values() for item in sublist]
        keys_list = [item for name in l.keys() for item in [name]*len(l[name])]
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
            if person not in self.know_faces[0] or person not in self.cache[0]:   
                pix = os.listdir(self.dataset_dir + person)
                
                for person_img in pix:
                    face = face_recognition.load_image_file(self.dataset_dir + person + "/" + person_img)
                    shape = face.shape
                    face_bounding_boxes = face_recognition.face_locations(face, model='yolov8')

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
                        
                        # Database saving service
                        data = [person, list(face_enc)]
                        try:
                            if person not in self.cache[0]:
                                self.saveToDB(data)
                        except rospy.ServiceException as e:
                            rospy.logerr('Service call failed: %s' % e)

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

    def _encapsulateDataRequest(self, data):
        h = Header()
        h.stamp = rospy.Time.now()
        h.seq = self.seq
        
        description = FaceDescription()
        description.header = h
    
        description.label = data[0]
        description.encoding = data[1]
        
        request = FaceEncoding()
        request.header = h
        request.descriptions.append(description)
        
        return request
    
    def saveToDB(self, data):
        rospy.wait_for_service(self.cache_writer_servername)
        try:
            self.cache_writer_service = rospy.ServiceProxy(self.cache_writer_servername, RedisCacheWriterSrv)
            request = self._encapsulateDataRequest(data)
            response = self.cache_writer_service(request)
            if response.response:
                rospy.loginfo('New face was saved in database.')
                self.getCache()
                
            else:
                rospy.logerr('New face was not saved in database.')
        except rospy.ServiceException as e:
            rospy.logerr("Service call failed: %s" % e)

    def getCache(self):
        rospy.wait_for_service(self.cache_reader_servername)
        try:
            self.cache_reader = rospy.ServiceProxy(self.cache_reader_servername, RedisCacheReaderSrv)
            cache = self._saveCache(self.cache_reader(Empty()))
        except rospy.ServiceException as e:
            cache = {}
            rospy.logerr("Service call failed: %s" % e)
            
        return cache
    
    def _saveCache(self, response):
        cache = {}
        cache_global_id = []
        
        data = response.response
        data_header = data.header
        descriptions = data.descriptions
        
        for item in descriptions:
            h = item.header
            if item.label not in cache.keys():
                cache[item.label] = []
            cache[item.label].append(list(item.encoding))
            cache_global_id.append(item.global_id)
        if cache != {}:
            rospy.loginfo('Cache was saved and is not empty.')
        #print(cache.keys())
        formated_cache = self.flatten(cache)
        formated_cache_final = formated_cache + tuple(cache_global_id)
        return formated_cache_final
    
    def PeopleIntroducing(self, ros_srv):

        name = ros_srv.name
        num_images = ros_srv.num_images
        NAME_DIR = os.path.join(self.dataset_dir, name)
        
        if os.path.exists(NAME_DIR):
            NAME_DIR = os.path.join(self.dataset_dir, name + '_1')
        
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
                ros_image_aux = rospy.wait_for_message(self.subscribers_dict['image_rgb'], Image, 1000)
            except (ROSException, ROSInterruptException) as e:
                break
            ros_image = ros_numpy.numpify(ros_image_aux)
            ros_image = np.flip(ros_image)
            ros_image = np.flipud(ros_image)
            
            face_locations = face_recognition.face_locations(ros_image, model='yolov8')
                
            if len(face_locations) > 0:
                rospy.logwarn('Picture ' + add_image_labels[i] + ' was  saved.')
                cv2.imwrite(os.path.join(NAME_DIR, add_image_labels[i]), ros_image)
                ros_image = cv2.cvtColor(ros_image, cv2.COLOR_BGR2RGB)
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
            global_id = ""
            if(len(self.cache[0]) > 0):
                #rospy.loginfo('Using cache')
                face_distances = np.linalg.norm(self.cache[1] - current_encoding, axis = 1)
                #print(self.cache[0])
                min_distance_idx = np.argmin(face_distances)
                min_distance = face_distances[min_distance_idx]
                if min_distance < thold:
                    name = (self.cache[0][min_distance_idx])
                    global_id = self.cache[2][min_distance_idx]
            elif(len(self.know_faces[0]) > 0):
                #rospy.loginfo('Using pkl file')
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
            description.global_id = global_id
            description.type = Description2D.DETECTION
            description.id = description.header.seq
            description.score = 1
            description.max_size = Vector3(*[0.2, 0.2, 0.2])
            size = int(right-left), int(bottom-top)
            description.bbox.center.x = int(left) + int(size[1]/2)
            description.bbox.center.y = int(top) + int(size[0]/2)
            description.bbox.size_x = bottom-top
            description.bbox.size_y = right-left

            cv2.rectangle(debug_img, (left, top), (right, bottom), (0, 255, 0), 2)
            
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(debug_img, name, (left + 4, bottom - 4), font, 0.5, (0,0,255), 2)
            description_header.seq += 1

            face_rec.descriptions.append(description)
            
        self.debug_publisher.publish(ros_numpy.msgify(Image, debug_img, 'bgr8'))
        if len(face_rec.descriptions) > 0:
            self.face_recognition_publisher.publish(face_rec)

    def readParameters(self):
        self.debug_topic = rospy.get_param("~publishers/debug/topic", "/butia_vision/br/debug")
        self.debug_qs = rospy.get_param("~publishers/debug/queue_size", 1)

        self.face_recognition_topic = rospy.get_param("~publishers/face_recognition/topic", "/butia_vision/br/face_recognition")

        self.face_recognition_qs = rospy.get_param("~publishers/face_recognition/queue_size", 1)

        self.introduct_person_servername = rospy.get_param("~servers/introduct_person/servername", "/butia_vision/br/introduct_person")
        
        self.cache_writer_servername = "redis_cache_writer_srv"
        
        self.cache_reader_servername = "redis_cache_reader_srv"

        super().readParameters()
        rospy.loginfo('foi 1')


if __name__ == '__main__':
    rospy.init_node('face_recognition_node', anonymous = True)
    
    face_rec = FaceRecognition()

    rospy.spin()
