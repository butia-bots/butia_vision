#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import rospy

import rospkg

from sensor_msgs.msg import Image
from std_srvs.srv import Empty, EmptyResponse

from butia_vision_msgs.srv import ListClasses, ListClassesResponse

from butia_vision_bridge import VisionSynchronizer

def ifState(func):
    def wrapper_func(*args, **kwargs):
        if args[0].state:
            func(*args, **kwargs)
    return wrapper_func


class BaseRecognition:
    def __init__(self, state=True):
        self.readParameters()

        self.rospack = rospkg.RosPack().get_path('butia_generic_recognition')

        self.state = state

    def initRosComm(self):
        self.recognized_objects_pub = rospy.Publisher(self.object_recognition_topic, Recognitions2D, queue_size=self.object_recognition_qs)

        self.list_objects_server = rospy.Service(self.list_objects_service, ListClasses, self.getObjectList)

        self.start_server = rospy.Service(self.start_service, Empty, self.start)

        self.stop_server = rospy.Service(self.stop_service, Empty, self.stop)

    def start(self, req):
        self.state = True
        return EmptyResponse()

    def stop(self, req):
        self.state = False
        return EmptyResponse()

    def getObjectList(self, req):
        objects_list = []
        for key, value in self.classes_by_category.items():
            for elem in value:
                objects_list.append(key + '/' + elem)
        return ListClassesResponse(objects_list)

    def yoloRecognitionCallback(self, img):

        if self.state == True:

            with torch.no_grad():

                rospy.loginfo('Image ID: ' + str(img.header.seq))

                cv_img = np.frombuffer(img.data, dtype=np.uint8).reshape(img.height, img.width, -1)

                #cv_img = self.bridge.imgmsg_to_cv2(img, desired_encoding='rgb8').copy()

                #results = self.model(torch.tensor(cv_img.reshape((1, 3, 640, 480)).astype(np.float32)).to(self.model.device))
                results = self.model(cv_img)

                bbs_l = results.pandas().xyxy[0]

                objects = []
                people = []

                for i in range(len(bbs_l)):

                    if int(bbs_l['class'][i]) >= len(self.all_classes):
                        continue
                    bbs_l['name'][i] = self.all_classes[int(bbs_l['class'][i])]
                    reference_model = bbs_l['name'][i]
                    print(bbs_l['name'][i])
                    #cv_img = cv2.cvtColor(cv_img, cv2.COLOR_RGB2BGR)
                    cv_img = cv2.rectangle(cv_img, (int(bbs_l['xmin'][i]), int(bbs_l['ymin'][i])), (int(bbs_l['xmax'][i]), int(bbs_l['ymax'][i])), self.colors[bbs_l['name'][i]])
                    cv_img = cv2.putText(cv_img, bbs_l['name'][i], (int(bbs_l['xmin'][i]), int(bbs_l['ymin'][i])), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color=self.colors[bbs_l['name'][i]])
                    if ('people' in self.all_classes and bbs_l['name'][i] in self.classes_by_category['people'] or 'people' in self.all_classes and bbs_l['name'][i] == 'people') and bbs_l['confidence'][i] >= self.threshold:

                        person = Description()
                        person.label_class = 'people' + '/' + bbs_l['name'][i]
                        person.reference_model = reference_model
                        person.probability = bbs_l['confidence'][i]
                        person.bounding_box.minX = int(bbs_l['xmin'][i])
                        person.bounding_box.minY = int(bbs_l['ymin'][i])
                        person.bounding_box.width = int(bbs_l['xmax'][i] - bbs_l['xmin'][i])
                        person.bounding_box.height = int(bbs_l['ymax'][i] - bbs_l['ymin'][i])
                        people.append(person)

                    elif (bbs_l['name'][i] in [val for sublist in self.all_classes for val in sublist] or bbs_l['name'][i] in self.all_classes) and bbs_l['confidence'][i] >= self.threshold:

                        object_d = Description()
                        index = None
                        j = 0
                        for value in self.classes_by_category.values():
                            if bbs_l['name'][i] in value:
                                index = j
                            j += 1
                        object_d.reference_model = reference_model
                        object_d.label_class = self.all_classes[index] + '/' + bbs_l['name'][i] if index is not None else bbs_l['name'][i]

                        object_d.probability = bbs_l['confidence'][i]
                        object_d.bounding_box.minX = int(bbs_l['xmin'][i])
                        object_d.bounding_box.minY = int(bbs_l['ymin'][i])
                        object_d.bounding_box.width = int(bbs_l['xmax'][i] - bbs_l['xmin'][i])
                        object_d.bounding_box.height = int(bbs_l['ymax'][i]- bbs_l['ymin'][i])
                        objects.append(object_d)

                #cv2.imshow('YoloV5', cv_img)
                #cv2.waitKey(1)

                debug_msg = Image()
                debug_msg.data = cv_img[:,:,::-1].flatten().tolist()
                debug_msg.encoding = "rgb8"
                debug_msg.width = cv_img.shape[1]
                debug_msg.height = cv_img.shape[0]
                debug_msg.step = debug_msg.width*3
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

        self.all_classes = list(rospy.get_param("/object_recognition/all_classes"))

        self.classes_by_category = dict(rospy.get_param("/object_recognition/classes_by_category"))

        self.model_file = rospy.get_param("/object_recognition/model_file", "larc2021_go_and_get_it.pt")