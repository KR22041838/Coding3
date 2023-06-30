#!/usr/bin/env python
# coding: utf-8

# In[15]:


import cv2
import matplotlib.pyplot as plt
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np

MARGIN = 10  # pixels
FONT_SIZE = 5
FONT_THICKNESS = 10
HANDEDNESS_TEXT_COLOR = (88, 205, 54)  # vibrant green
LANDMARK_CIRCLE_RADIUS = 10  # increase this value for thicker landmarks
CONNECTION_LINE_THICKNESS = 10  # increase this value for thicker connection lines

def draw_landmarks_on_image(rgb_image, detection_result):
    hand_landmarks_list = detection_result.hand_landmarks
    handedness_list = detection_result.handedness
    annotated_image = np.copy(rgb_image)

    # Loop through the detected hands to visualize.
    for idx in range(len(hand_landmarks_list)):
        hand_landmarks = hand_landmarks_list[idx]
        handedness = handedness_list[idx]

        # Draw the hand landmarks.
        hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        hand_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
        ])
        solutions.drawing_utils.draw_landmarks(
            annotated_image,
            hand_landmarks_proto,
            solutions.hands.HAND_CONNECTIONS,
            landmark_drawing_spec=solutions.drawing_styles.DrawingSpec(
                color=(0, 255, 0),  # Use a custom color if desired
                thickness=LANDMARK_CIRCLE_RADIUS,
                circle_radius=LANDMARK_CIRCLE_RADIUS
            ),
            connection_drawing_spec=solutions.drawing_styles.DrawingSpec(
                color=(0, 0, 255),  # Use a custom color if desired
                thickness=CONNECTION_LINE_THICKNESS
            )
        )

        # Draw handedness (left or right hand) on the image.
        height, width, _ = annotated_image.shape
        x_coordinates = [landmark.x for landmark in hand_landmarks]
        y_coordinates = [landmark.y for landmark in hand_landmarks]
        text_x = int(min(x_coordinates) * width)
        text_y = int(min(y_coordinates) * height) - MARGIN

        cv2.putText(
            annotated_image,
            f"{handedness[0].category_name}",
            (text_x, text_y),
            cv2.FONT_HERSHEY_DUPLEX,
            FONT_SIZE,
            HANDEDNESS_TEXT_COLOR,
            FONT_THICKNESS,
            cv2.LINE_AA
        )

    return annotated_image



# In[18]:


# STEP 1: Import the necessary modules.
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
import matplotlib.pyplot as plt

# STEP 2: Create an HandLandmarker object.
base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(base_options=base_options,
                                       num_hands=2)
detector = vision.HandLandmarker.create_from_options(options)
# STEP 3: Load the input image.
image = mp.Image.create_from_file("signAt1.jpg")

# STEP 4: Detect hand landmarks from the input image.
detection_result = detector.detect(image)




# In[19]:


# STEP 5: Process the classification result. In this case, visualize it.
annotated_image = draw_landmarks_on_image(image.numpy_view(), detection_result)
annotated_image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)

# Display the annotated image
plt.imshow(annotated_image_rgb)
plt.axis('off')
plt.show()


print(detection_result.hand_landmarks)


# In[ ]:





# In[4]:


# STEP 3: Load the input image.
image = mp.Image.create_from_file("signA.jpg")

# STEP 4: Detect hand landmarks from the input image.
detection_result = detector.detect(image)



# In[ ]:




