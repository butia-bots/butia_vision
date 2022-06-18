#!/usr/bin/env python3
import rospy

from std_msgs.msg import Header
from std_srvs.srv import Empty, EmptyResponse
import torch
import torchvision
import cv2
import cv_bridge
import numpy as np
import rospkg
import os
from sensor_msgs.msg import Image
from butia_vision_msgs.msg import Description, Recognitions
from butia_vision_msgs.srv import ListClasses, ListClassesResponse

torch.set_num_threads(1)

dictionary={"person_standing": "Person",
            "ycb_002_master_chef_can": "master_chef_can", 
            "ycb_003_cracker_box": "cracker_box", 
            "ycb_004_sugar_box": "sugar_box", 
            "ycb_005_tomato_soup_can": "tomato_soup_can", 
            "ycb_006_mustard_bottle": "mustard_bottle", 
            "ycb_007_tuna_fish_can": "tuna_fish_can", 
            "ycb_008_pudding_box": "pudding_box", 
            "ycb_009_gelatin_box": "gelatin_box", 
            "ycb_010_potted_meat_can": "potted_meat_can", 
            "ycb_011_banana": "banana", 
            "ycb_012_strawberry": "strawberry", 
            "ycb_013_apple": "apple", 
            "ycb_014_lemon": "lemon", 
            "ycb_015_peach": "peach", 
            "ycb_016_pear": "pear", 
            "ycb_017_orange": "orange", 
            "ycb_018_plum": "plum", 
            "ycb_019_pitcher_base": "Pitcher base", 
            "ycb_021_bleach_cleanser": "Srub cleanser bottle", 
            "ycb_022_windex_bottle": "Windex Spray bottle", 
            "ycb_024_bowl": "Bowl", 
            "ycb_025_mug": "Mug", 
            "ycb_026_sponge": "Scotch brite dobie sponge", 
            "ycb_029_plate": "Plate", 
            "ycb_030_fork": "Fork", 
            "ycb_031_spoon": "Spoon", 
            "ycb_033_spatula": "Spatula", 
            "ycb_040_large_marker": "Large marker", 
            "ycb_051_large_clamp": "Clamps", 
            "ycb_053_mini_soccer_ball": "Mini soccer ball", 
            "ycb_054_softball": "Soft ball", 
            "ycb_055_baseball": "Baseball", 
            "ycb_056_tennis_ball": "Tennis ball", 
            "ycb_057_racquetball": "Racquetball", 
            "ycb_058_golf_ball": "Golf ball", 
            "ycb_059_chain": "Chain", 
            "ycb_061_foam_brick": "Foam brick", 
            "ycb_062_dice": "Dice", 
            "ycb_063-a_marbles": "Marbles", 
            "ycb_063-b_marbles": "Marbles", 
            "ycb_065-a_cups": "Cups", 
            "ycb_065-b_cups": "Cups", 
            "ycb_065-c_cups": "Cups", 
            "ycb_065-d_cups": "Cups", 
            "ycb_065-e_cups": "Cups", 
            "ycb_065-f_cups": "Cups", 
            "ycb_065-g_cups": "Cups", 
            "ycb_065-h_cups": "Cups", 
            "ycb_065-i_cups": "Cups", 
            "ycb_065-j_cups": "Cups", 
            "ycb_070-a_colored_wood_blocks": "Colored wood blocks", 
            "ycb_070-b_colored_wood_blocks": "Colored wood blocks", 
            "ycb_071_nine_hole_peg_test": "9-peg-hole test", 
            "ycb_072-a_toy_airplane": "Toy airplane", 
            "ycb_072-b_toy_airplane": "Toy airplane", 
            "ycb_072-c_toy_airplane": "Toy airplane", 
            "ycb_072-d_toy_airplane": "Toy airplane", 
            "ycb_072-e_toy_airplane": "Toy airplane", 
            "ycb_073-a_lego_duplo": "Lego duplo", 
            "ycb_073-b_lego_duplo": "Lego duplo", 
            "ycb_073-c_lego_duplo": "Lego duplo", 
            "ycb_073-d_lego_duplo": "Lego duplo", 
            "ycb_073-e_lego_duplo": "Lego duplo", 
            "ycb_073-f_lego_duplo": "Lego duplo", 
            "ycb_073-g_lego_duplo": "Lego duplo", 
            "ycb_077_rubiks_cube": "Rubik’s cube"
}

colors = dict([(k, np.random.randint(low=0, high=256, size=(3,)).tolist()) for k,v in dictionary.items()])


class YoloRecognition():
    def __init__(self):
        self.readParameters()

        self.rospack = rospkg.RosPack()
        print(self.rospack.get_path('object_recognition'))

        #self.bounding_boxes_sub = rospy.Subscriber(self.bounding_boxes_topic, BoundingBoxes, self.yoloRecognitionCallback, queue_size=self.bounding_boxes_qs)
        self.bridge = cv_bridge.CvBridge()
        self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=os.path.join(self.rospack.get_path('object_recognition'), 'yolov5_network_config', 'weights', self.model_file), autoshape=True)
        self.model.eval()
        print('Done loading model!')
        self.image_sub = rospy.Subscriber(self.image_topic, Image, self.yoloRecognitionCallback, queue_size=self.image_qs)
        self.debug_image_pub = rospy.Publisher(self.debug_image_topic, Image, queue_size=self.debug_image_qs)

        self.recognized_objects_pub = rospy.Publisher(self.object_recognition_topic, Recognitions, queue_size=self.object_recognition_qs)
        self.recognized_people_pub = rospy.Publisher(self.people_detection_topic, Recognitions, queue_size=self.people_detection_qs)
        self.object_list_updated_pub = rospy.Publisher(self.object_list_updated_topic, Header, queue_size=self.object_list_updated_qs)

        self.list_objects_server = rospy.Service(self.list_objects_service, ListClasses, self.getObjectList)

        self.start_server = rospy.Service(self.start_service, Empty, self.start)

        self.stop_server = rospy.Service(self.stop_service, Empty, self.stop)

        self.object_list_updated_header = Header()
        self.object_list_updated_header.stamp = rospy.get_rostime().now
        self.object_list_updated_header.frame_id = "objects_list"

        self.object_list_updated_pub.publish(self.object_list_updated_header)

        self.state = False

    def start(self, req):
        self.state = True
        return EmptyResponse()

    def stop(self, req):
        self.state = False
        return EmptyResponse()
    
    def getObjectList(self, req):
        objects_list = []
        for key, value in self.possible_classes.items():
            for elem in value:
                objects_list.append(key + '/' + elem)
        return ListClassesResponse(objects_list)

    def yoloRecognitionCallback(self, img):

        if self.state == True:

            with torch.no_grad():

                rospy.loginfo('Image ID: ' + str(img.header.seq))

                cv_img = self.bridge.imgmsg_to_cv2(img, desired_encoding='rgb8').copy()

                #results = self.model(torch.tensor(cv_img.reshape((1, 3, 640, 480)).astype(np.float32)).to(self.model.device))
                results = self.model(cv_img)

                bbs_l = results.pandas().xyxy[0]

                objects = []
                people = []

                for i in range(len(bbs_l)):
                    print(bbs_l['name'][i])
                    cv_img = cv2.cvtColor(cv_img, cv2.COLOR_RGB2BGR)
                    cv_img = cv2.rectangle(cv_img, (int(bbs_l['xmin'][i]), int(bbs_l['ymin'][i])), (int(bbs_l['xmax'][i]), int(bbs_l['ymax'][i])), colors[bbs_l['name'][i]])
                    cv_img = cv2.putText(cv_img, bbs_l['name'][i], (int(bbs_l['xmin'][i]), int(bbs_l['ymin'][i])), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color=colors[bbs_l['name'][i]])
                    if bbs_l['name'][i] in dictionary.keys():
                        reference_model = bbs_l['name'][i]
                        bbs_l['name'][i] = dictionary[bbs_l['name'][i]]
                        
                    if 'people' in self.possible_classes and bbs_l['name'][i] in self.possible_classes['people'] and bbs_l['confidence'][i] >= self.threshold:
                        person = Description()
                        person.label_class = 'people' + '/' + bbs_l['name'][i]
                        person.reference_model = reference_model
                        person.probability = bbs_l['confidence'][i]
                        person.bounding_box.minX = int(bbs_l['xmin'][i])
                        person.bounding_box.minY = int(bbs_l['ymin'][i])
                        person.bounding_box.width = int(bbs_l['xmax'][i] - bbs_l['xmin'][i])
                        person.bounding_box.height = int(bbs_l['ymax'][i] - bbs_l['ymin'][i])
                        people.append(person)

                    elif bbs_l['name'][i] in [val for sublist in self.possible_classes.values() for val in sublist] and bbs_l['confidence'][i] >= self.threshold:
                        object_d = Description()
                        index = 0
                        j = 0
                        for value in self.possible_classes.values():
                            if bbs_l['name'][i] in value:
                                index = j
                            j += 1
                        object_d.reference_model = reference_model
                        object_d.label_class = list(self.possible_classes.keys())[index] + '/' + bbs_l['name'][i]
                        object_d.probability = bbs_l['confidence'][i]
                        object_d.bounding_box.minX = int(bbs_l['xmin'][i])
                        object_d.bounding_box.minY = int(bbs_l['ymin'][i])
                        object_d.bounding_box.width = int(bbs_l['xmax'][i] - bbs_l['xmin'][i])
                        object_d.bounding_box.height = int(bbs_l['ymax'][i]- bbs_l['ymin'][i])
                        objects.append(object_d)

                #cv2.imshow('YoloV5', cv_img)
                #cv2.waitKey(1)

                debug_msg = self.bridge.cv2_to_imgmsg(cv_img)
                self.debug_image_pub.publish(debug_msg)

                objects_msg = Recognitions()
                people_msg = Recognitions()

                if len(objects) > 0:
                    #objects_msg.header = bbs.header
                    objects_msg.image_header = img.header
                    objects_msg.descriptions = objects
                    self.recognized_objects_pub.publish(objects_msg)

                if len(people) > 0:
                    #people_msg.header = bbs.header
                    people_msg.image_header = img.header
                    people_msg.descriptions = people
                    self.recognized_people_pub.publish(people_msg)
            


    def readParameters(self):
        self.bounding_boxes_topic = rospy.get_param("/object_recognition/subscribers/bounding_boxes/topic", "/darknet_ros/bounding_boxes")
        self.bounding_boxes_qs = rospy.get_param("/object_recognition/subscribers/bounding_boxes/queue_size", 1)

        self.image_topic = rospy.get_param("/object_recognition/subscribers/image/topic", "/butia_vision/bvb/image_rgb_raw")
        self.image_qs = rospy.get_param("/object_recognition/subscribers/image/queue_size", 1)

        self.debug_image_topic = rospy.get_param("/object_recognition/publishers/debug_image/topic", "/butia_vision/or/debug")
        self.debug_image_qs = rospy.get_param("/object_recognition/publishers/debug_image/queue_size", 1)

        self.object_recognition_topic = rospy.get_param("/object_recognition/publishers/object_recognition/topic", "/butia_vision/or/object_recognition")
        self.object_recognition_qs = rospy.get_param("/object_recognition/publishers/object_recognition/queue_size", 1)

        self.people_detection_topic = rospy.get_param("/object_recognition/publishers/people_detection/topic", "/butia_vision/or/people_detection")
        self.people_detection_qs = rospy.get_param("/object_recognition/publishers/people_detection/queue_size", 1)

        self.object_list_updated_topic = rospy.get_param("/object_recognition/publishers/object_list_updated/topic", "/butia_vision/or/object_list_updated")
        self.object_list_updated_qs = rospy.get_param("/object_recognition/publishers/object_list_updated/queue_size", 1)

        self.list_objects_service = rospy.get_param("/object_recognition/servers/list_objects/service", "/butia_vision/or/list_objects")
        self.start_service = rospy.get_param("/object_recognition/servers/start/service", "/butia_vision/or/start")
        self.stop_service = rospy.get_param("/object_recognition/servers/stop/service", "/butia_vision/or/stop")

        self.threshold = rospy.get_param("/object_recognition/threshold", 0.5)
        
        self.possible_classes = dict(rospy.get_param("/object_recognition/possible_classes"))

        self.model_file = rospy.get_param("/object_recognition/model_file", "larc2021_go_and_get_it.pt")
