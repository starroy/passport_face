import cv2
import dlib
import numpy as np
from PIL import Image
import os

input_folder = 'input'
output_folder = 'output'

image_names = os.listdir(output_folder)
for image_name in image_names:
    image_path = os.path.join(output_folder, image_name)
    image = cv2.imread(image_path)
    # image = Image.open(os.path.join(output_folder, image_name))
    # image.show()
    # Initialize face detector and shape predictor
    
    
    # detector = dlib.get_frontal_face_detector()
    # predictor_path = 'face_detect/shape_predictor_81_face_landmarks.dat'
    # predictor = dlib.shape_predictor(predictor_path)

    # # Convert the image to grayscale
    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # # Detect faces in the grayscale image
    # faces = detector(gray)

    # # Iterate over the detected faces
    # for face in faces:
    #     # Predict the facial landmarks
    #     shape = predictor(gray, face)
    #     landmarks = np.matrix([[p.x, p.y] for p in shape.parts()])

    #     # Find the bounding rectangle of the face
    #     x, y, w, h = cv2.boundingRect(landmarks)

    #     # Draw a rectangle around the face
    #     cv2.rectangle(image, (int(x-0.7*w), int(y-0.3*h)), (int(1.3*w) + x, int(1.3*h) + y), (0, 255, 0), 2)

    #     # Find the position of the eyes
    #     left_eye = landmarks[36]
    #     right_eye = landmarks[45]

    #     # Draw circles on the eyes
    #     cv2.circle(image, (left_eye[0, 0], left_eye[0, 1]), 3, (0, 0, 255), -1)
    #     cv2.circle(image, (right_eye[0, 0], right_eye[0, 1]), 3, (0, 0, 255), -1)


    #     # Find the position of the thread
    #     thread= landmarks[8]

    #     # Draw a circle on the thread
    #     cv2.circle(image, (thread[0, 0], thread[0, 1]), 3, (0, 0, 255), -1)
    #     min_y = min(landmarks[:, 1])
    #     print (thread, left_eye, right_eye)
    #     print (x, y, w, h)
        

    # # Display the image with the face and eyes highlighted
    # cv2.imwrite('Face and Eyes'+image_name, image)
    
    
    detector = dlib.get_frontal_face_detector()
    predictor_path = 'face_detect/shape_predictor_81_face_landmarks.dat'
    predictor = dlib.shape_predictor(predictor_path)

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale image
    faces = detector(gray)

    # Iterate over the detected faces
    for face in faces:
        # Predict the facial landmarks
        shape = predictor(gray, face)
        landmarks = np.matrix([[p.x, p.y] for p in shape.parts()])

        # Find the bounding rectangle of the face
        x, y, w, h = cv2.boundingRect(landmarks)

        # Draw a rectangle around the face
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Find the position of the eyes
        left_eye = landmarks[36]
        right_eye = landmarks[45]

        # Draw circles on the eyes
        cv2.circle(image, (left_eye[0, 0], left_eye[0, 1]), 3, (0, 0, 255), -1)
        cv2.circle(image, (right_eye[0, 0], right_eye[0, 1]), 3, (0, 0, 255), -1)

        # Find the position of the thread
        thread = landmarks[8]

        # Draw a circle on the thread
        cv2.circle(image, (thread[0, 0], thread[0, 1]), 3, (0, 0, 255), -1)

        # Calculate the dimensions of the head area
        min_y = min(landmarks[:, 1])
        head_height = h - min_y
        head_width = w * 1.5

        # Calculate the distance between the eyes
        distance_between_eyes = np.sqrt((left_eye[0, 0] - right_eye[0, 0])**2 + (left_eye[0, 1] - right_eye[0, 1])**2)

        # Resize the head area
        resized_head_height = int(head_height * 2)
        resized_head_width = int(head_width * 2)
        resized_head_area = cv2.resize(image[y:y + h, x:x + w], (resized_head_width, resized_head_height))
        image[y:y + h, x:x + w] = resized_head_area

    # Display the image with the face and eyes highlighted
    cv2.imwrite('output_image.jpg', image)

