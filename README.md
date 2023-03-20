# butia_vision

## Overview
This is a group of ROS packages responsable for perform computer vision process of Butia Bots domestic robot (DoRIS) in Robocup@Home league. 

**Author: [Igor Maurell], igormaurell@furg.br**
**Author: [Miguel Martins], migueldossantos@furg.br**

## Dependencies
This software is built on the Robotic Operating System ([ROS]), which needs to be [installed](http://wiki.ros.org) first. Additionally, the packages depends of a few libraries and frameworks:

- [OpenCV](http://opencv.org/) (computer vision library)
- [Openface](https://cmusatyalab.github.io/openface/) (face recognition library)
- [scikit-learn](http://scikit-learn.org/stable/) (machine learning library)
- [darknet_ros](https://github.com/leggedrobotics/darknet_ros) (darknet ros package)
- [mask_rcnn_ros](https://github.com/crislmfroes/mask_rcnn_ros)

## Packages
The vision system has three main packages and other four helpers.

### Main Packages
- object-recognition
- face_recognition
- people_tracking

### Helper Packages
- butia_vision_bridge
- butia_vision_msgs
- segmentation
- image2kinect
- image_server

## Instalation
Before run catkin_make command, is adiviseble to run the follow commands:

	```
	chmod +x install.sh
	```
	To execute the install file:
	```
	sudo ./install.sh
	```