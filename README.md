# butia_vision

## Overview
This is a group of ROS packages responsable for perform computer vision process of [BUTIÁBots](https://fbot.vercel.app/) domestic robot (DoRIS) in [Robocup@Home](https://athome.robocup.org/) league. 

**Author: [Igor Maurell], igormaurell@furg.br**
**Author: [Miguel Martins], migueldossantos@furg.br**

## Dependencies
This software is built on the Robotic Operating System ([ROS Noetic]), which needs to be [installed](https://github.com/butia-bots/butia_learning/wiki/Instala%C3%A7%C3%B5es-importantes#ros-robot-operating-system) first. Additionally, the packages depends of a few libraries and frameworks:

- [OpenCV](http://opencv.org/) (computer vision library);
- [Openface](https://cmusatyalab.github.io/openface/) (face recognition library);
- [scikit-learn](http://scikit-learn.org/stable/) (machine learning library);
- [darknet_ros](https://github.com/leggedrobotics/darknet_ros) (darknet ros package);
- [mask_rcnn_ros](https://github.com/crislmfroes/mask_rcnn_ros).

## Packages
The vision system has main packages, helpers and 3rd helpers.

### Main Packages
- object_recognition;
- face_recognition;
- people_tracking.

### Helper Packages
- butia_vision_bridge;
- butia_vision_msgs;
- image2kinect.

### Helper 3rd Packages
- [iai_kinect2](https://github.com/butia-bots/iai_kinect2);
- [libfreenect2](https://github.com/butia-bots/libfreenect2);
- [yolov5](https://github.com/butia-bots/yolov5).

## Clone

Clone this repository using the follow command:
```
git clone --recursive https://github.com/butia-bots/butia_vision.git
```

## Instalation
Before run catkin_make command, is adiviseble to run the follow commands:

	```
	chmod +x install.sh
	```
	To execute the install file:
	```
	sudo ./install.sh
	```
The script will created a folder named "butia_ws" to be the workspace