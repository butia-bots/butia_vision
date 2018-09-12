#! /usr/bin/env python

class FaceEmbosser():
    def __init__(self, embosser_lib = 'facenet'):
        self.embosser_lib = embosser_lib

        self.embossers_dict = {}

    def loadFacenetModels(self):
        self.net = openface.TorchNeuralNet(os.path.join(self.models_dir, 'openface', self.openface_model), self.image_dimension, cuda = self.cuda)
        self.embossers_dict['facenet'] = extractFeaturesFacenet

    def extractFeaturesFacenet(self, image):
        #now_s = rospy.get_rostime().to_sec()
        features = self.net.forward(image)
        #rospy.loginfo("Feature extraction took: " + str(rospy.get_rostime().to_sec() - now_s) + " seconds.")
        return features

    def extractFeatures(self, image):
        try:
            features = self.embossers_dict[self.embosser_lib](image)
        except KeyError:
            print(self.embosser_lib + ' model is not loaded.')
            return None
        return features