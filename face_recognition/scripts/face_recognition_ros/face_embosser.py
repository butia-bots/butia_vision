#! /usr/bin/env python

class FaceEmbosser():
    def __init__(self):

    def loadTorchNeuralNetModel(self):
        self.net = openface.TorchNeuralNet(os.path.join(self.models_dir, 'openface', self.openface_model), self.image_dimension, cuda = self.cuda)

    def extractFeatures(self, image):
        now_s = rospy.get_rostime().to_sec()
        feature_vector = self.net.forward(image)
        rospy.loginfo("Feature extraction took: " + str(rospy.get_rostime().to_sec() - now_s) + " seconds.")
        return feature_vector