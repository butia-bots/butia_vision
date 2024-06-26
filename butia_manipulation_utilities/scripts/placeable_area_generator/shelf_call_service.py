#!/usr/bin/env python3

import rospy
from butia_vision_msgs.msg import Recognitions3D  
from butia_vision_msgs.srv import PlaceableArea, PlaceableAreaRequest 
from butia_vision_msgs.srv import ShelfClassificationRequest, ShelfClassification

class Listener:
    def __init__(self):
        self.flag = True
        self.listener()
        self.MarkerMsgs = None

    def callbackPoints(self, msg):
        # Call the service
        print('oi')
        if self.flag:
            rospy.loginfo("Received message!")
            self.flag = False
            self.call_service(msg, self.MarkerMsgs)
            self.flag = False
    def callbackMarker(self, msg: Recognitions3D): 
        # Call the service
        try:
            # Create a service proxy
            service_proxy = rospy.ServiceProxy('/butia_vision_msgs/shelf_classification', ShelfClassification)
            
            # Create a request object
            request = ShelfClassificationRequest()
            
            # Set the message as an argument in the request
            request.objects = msg
            request.labelObjectToPut = msg.markers[-1]

            # Save the markers to a file
            with open('./message.txt', 'w') as file:
                file.write(str(msg))

            
            # Call the service
            response = service_proxy(request)
            
            rospy.loginfo("Service call successful: %s", response)
        
        except rospy.ServiceException as e:
            rospy.logerr("Service call failed: %s", e)
        
    def listener(self):
        # Subscribe to the topic
        print('oi')
        rospy.Subscriber('/butia_vision/br/object_recognition3d', Recognitions3D, self.callbackMarker)

if __name__ == '__main__':
    rospy.init_node('listener_node', anonymous=False)
    listener = Listener()
    rospy.spin()
