
import face_recognition
import os
import cv2
import time
import rospkg
from openface import helper
import rospy

def regressiveCounter(sec):
    for i  in range(0, sec):
        print(str(sec-i) + '...')
        time.sleep(1)

def encode_faces(self, train_direct):

    train_direct = '/home/butiabots/Desktop/testes_joao/face_recognition/train_dir/'
    encodings = []
    names = []
    encoded_face = {}

    train_dir = os.listdir(train_direct)

    for person in train_dir:
        pix = os.listdir(train_direct + person)

        # Loop through each training image for the current person
        for person_img in pix:
            # Get the face encodings for the face in each image file
            face = face_recognition.load_image_file(train_direct + person + "/" + person_img)
            print(person)
            face_bounding_boxes = face_recognition.face_locations(face)

            #If training image contains exactly one face
            if len(face_bounding_boxes) == 1:
                face_enc = face_recognition.face_encodings(face)[0]
                # Add face encoding for current image with corresponding label (name) to the training data
                encodings.append(face_enc)
                #print (encodings)
                if person not in names:
                    names.append(person)
                    encoded_face[person] = []
                    encoded_face[person].append(face_enc)
                else:
                    encoded_face[person].append(face_enc)
            else:
                print(person + "/" + person_img + " was skipped and can't be used for training")

    return encoded_face

def peopleIntroducing(ros_srv):

    DATASET_DIR = os.path.join(rospkg.RosPack().get_path('face_recognition'), 'dataset')

    name = ros_srv.name
    num_images = ros_srv.num_images
    NAME_DIR = os.path.join(DATASET_DIR, 'raw', name)
    helper.mkdirP(NAME_DIR)

    image_type = '.jpg'

    image_labels = os.listdir(NAME_DIR)
    add_image_labels = []
    i = 1
    k = 0
    j = num_images
    number = [] 
    for label in image_labels:
        number.append(int(label.replace(image_type, '')))
    
    number.sort()
    n = 1
    while j > 0:
        if k < len(number):
            n = number[k] + 1
            if number[k] == i:
                k += 1
            else:
                add_image_labels.append((str(i) + image_type))
                j -= 1      
            i += 1 

        else:
            add_image_labels.append(str(n) + image_type)
            j -= 1
            n += 1
    
    num_images = ros_srv.num_images

    i = 0
    while i<num_images:
        try:
            ros_image = rospy.wait_for_message(image_topic, Image, 1000)
        except (ROSException, ROSInterruptException) as e:
            print(e)
            break

        face = face_recognition.face_locations(ros_image)
        s_rgb_image = ros_image.copy() 
        if face != None:
            if len(face):
                top, right, bottom, left = face[0]
                cv2.rectangle(s_rgb_image, (left, top), (right, bottom), (0, 0, 255), 2)

        cv2.imshow("Person", s_rgb_image)
        
        regressiveCounter(ros_srv.interval)

        if face != None:
            rospy.logwarn('Picture ' + add_image_labels[i] + ' was  saved.')
            cv2.imwrite(os.path.join(NAME_DIR, add_image_labels[i]), ros_image)
            i+= 1
        else:
            rospy.logerr("The face was not detected.")


    cv2.destroyAllWindows()

    return encode_faces()