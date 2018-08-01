import rospy
import openface
import cv2
import argparse
import json
import os
from cv_bridge import CvBridge

from sensor_msgs.msg import Image
from vision_system_msgs.msg import DetectedFaces

def imageListener(image_msg):
    cv_image = bridge.imgmsg_to_cv2(image_msg, desired_encoding="CV_8UC3")
    bb = align.getAllFaceBoundingBoxes(cv_image)
    print bb
    
#def publishDetection():

with open("../config/config.json") as config_file:
    config_data = json.load(config_file)

openface_dir = json.dumps("openface").dumps("dir")
dlibmodel_dir = os.path.join(openface_dir, 'models', 'dlib') 

parser = argparse.ArgumentParser()
parser.add_argument('--dlibFacePredictor', type=str, help="Path to dlib's face predictor.",
                    default=os.path.join(dlibmodel_dir, "shape_predictor_68_face_landmarks.dat"))

args = parser.parse_args()
align = openface.AlignDlib(args.dlibFacePredictor)

publisher = 0
bridge = CvBridge()

if __name__ == '__main__':
    rospy.init_node('face_detection_node', anonymous = True)

    rospy.Subscriber('/cv_camera/image_raw', Image, imageListener)

    publisher = rospy.Publisher('/vision_system/fr/face_detection', DetectedFaces, queue_size = 10)

    rospy.spin()