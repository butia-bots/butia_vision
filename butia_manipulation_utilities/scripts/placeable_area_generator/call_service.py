#!/usr/bin/env python3

import rospy
from butia_vision_msgs.msg import Recognitions3D  # Replace with your message type
from butia_vision_msgs.srv import PlaceableArea, PlaceableAreaRequest  # Replace with your service name and request type
from sensor_msgs.msg import PointCloud2
from visualization_msgs.msg import MarkerArray
class Listener:
    def __init__(self):
        self.flag = True
        self.listener()
        self.MarkerMsgs = None

    def callbackPoints(self, msg):
        # Call the service
        if self.flag:
            rospy.loginfo("Received message!")
            self.flag = False
            self.call_service(msg, self.MarkerMsgs)
            self.flag = False
    def callbackMarker(self, msg):  
        # Call the service
        if self.flag:
            self.MarkerMsgs = msg
            rospy.Subscriber('/butia_vision/bvb/points', PointCloud2, self.callbackPoints)

    def call_service(self, points, markers):
        try:
            # Create a service proxy
            service_proxy = rospy.ServiceProxy('/butia_vision_msgs/placeable_area', PlaceableArea)
            
            # Create a request object
            request = PlaceableAreaRequest()
            
            # Set the message as an argument in the request
            request.point_cloud = points
            request.markers = markers

            # Save the markers to a file
            with open('./message.txt', 'w') as file:
                file.write(str(markers))

            
            # Call the service
            response = service_proxy(request)
            
            rospy.loginfo("Service call successful: %s", response)
        
        except rospy.ServiceException as e:
            rospy.logerr("Service call failed: %s", e)

    def listener(self):
        # Subscribe to the topic
        rospy.Subscriber('/butia_vision/br/object_recognition_markers', MarkerArray, self.callbackMarker)

if __name__ == '__main__':
    rospy.init_node('listener_node', anonymous=False)
    listener = Listener()
    rospy.spin()

