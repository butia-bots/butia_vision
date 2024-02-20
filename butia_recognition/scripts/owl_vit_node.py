#!/usr/bin/env python3

import rospy
import ros_numpy

from PIL import Image
from PIL import ImageDraw
from copy import deepcopy
from transformers import pipeline
from butia_vision_msgs.msg import Recognitions2D, Description2D
from butia_recognition import BaseRecognition

class OwlVitRecognition(BaseRecognition):
    def __init__(self):
        self.detector = pipeline(model="google/owlvit-base-patch32", task="zero-shot-object-detection")
        self.message_pub = rospy.Publisher("butia_vision/br/recognition2D", Recognitions2D, queue_size=1)
        self.read_params()
        super().initRosComm(callbacks_obj=self)

    def read_params(self):
        self.prompt = rospy.get_param("~prompt", [])
        print("self.prompt: ", self.prompt)
        super().readParameters()

    def callback(self, *args):
        data = self.sourceDataFromArgs(args)

        img = data['image_rgb']
        img_depth = data['image_depth']
        camera_info = data['camera_info']
        HEADER = img.header

        recognition = Recognitions2D()
        recognition.image_rgb = img
        recognition.image_depth = img_depth
        recognition.camera_info = camera_info
        recognition.header = HEADER
        recognition.descriptions = []
        img = ros_numpy.numpify(img)

        debug_img = deepcopy(img)
        # results = None
        # bboxs = None

        # image = Image.open("imagem.jpg").convert("RGB")

        predictions = self.detector(
            img,
            candidate_labels=[self.prompt],
        )

        draw = ImageDraw.Draw(debug_img)
        recognition_msg = Recognitions2D()

        for i, prediction in enumerate(predictions):
            label = prediction['label']
            score = prediction['score']
            box = prediction['box']
            
            xmin, ymin, xmax, ymax = box.values() # ou box[0:4]

            draw.rectangle([(xmin, ymin), (xmax, ymax)], outline="red", width=2)
            
            draw.text((xmin, ymin), f"{label} ({score:.2f})", fill="red")

            description = Description2D
            description.header = HEADER
            description.id = i
            description.bbox.center.x = (xmin + xmax) / 2
            description.bbox.center.y = (ymin + ymax) / 2
            description.bbox.size_x = xmax - xmin
            description.bbox.size_y = ymax - ymin
            description.label = label
            description.score = score
            description.class_num = 'bag'

            recognition_msg.descriptions.append(description)

        self.message_pub.publish(recognition)

if __name__ == '__main__':
    rospy.init_node('owl_vit_node', anonymous = True)

    owl_vit = OwlVitRecognition()

    rospy.spin()
