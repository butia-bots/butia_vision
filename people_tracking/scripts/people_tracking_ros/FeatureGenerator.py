import cv2
import numpy as np
from cv_bridge import CvBridge
from sensor_msgs.msg import Image


class FeatureGenerator():
    
    def __init__(extractor,surf_param=400):
        if extractor == "surf":
            self.extractor = cv2.xfeatures2d.SURF_create(surf_param)
        else if extractor = "sift":
            self.extractor = cv2.xfeatures2d.SIFT_create()

    def extractFeatures(segmentedImages):
        features = []
        for img in segmentedImages:
            bw_img = cv2.cvtColor(img, cv2.RGB2GRAY)
            feature = {}
            feature["keypoints"], feature["descriptors"] = self.extractor.detectAndCompute(bw_img, None)
            features.append(feature)
        return features


            