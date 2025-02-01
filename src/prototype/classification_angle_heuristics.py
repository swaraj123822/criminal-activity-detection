# -*- coding: utf-8 -*-
"""
Created on Sun Feb  2 01:25:39 2025

@author: swara
"""

#%% imports
import cv2
import mediapipe as mp
import time
import numpy as np

#%%
# Function to calculate angle between three points
def calculate_angle(a, b, c):
    a = np.array(a)  # First point
    b = np.array(b)  # Mid point
    c = np.array(c)  # End point
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    
    if angle > 180.0:
        angle = 360 - angle
        
    return angle

#%%
# Function to classify human activity based on landmarks detected
def classify_action(landmarks):
    # Extract keypoints
    left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
    right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
    left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
    right_hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
    left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
    right_knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
    left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
    right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
    left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
    right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]

    # Calculate angles
    hip_angle_left = calculate_angle(left_shoulder, left_hip, left_knee)
    hip_angle_right = calculate_angle(right_shoulder, right_hip, right_knee)
    elbow_angle_left = calculate_angle(left_shoulder, left_elbow, left_wrist)
    elbow_angle_right = calculate_angle(right_shoulder, right_elbow, right_wrist)

    # Heuristic-based classification
    if hip_angle_left < 90 and hip_angle_right < 90:
        return 'Sitting'
    elif hip_angle_left > 160 and hip_angle_right > 160:
        return 'Standing'
    elif elbow_angle_left < 90 or elbow_angle_right < 90:
        return 'Punching'
    else:
        return 'Running'

#%%
# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose_video = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, model_complexity=1)
my_drawing = mp.solutions.drawing_utils

# Function to detect pose and classify action
def detectPose(image, pose, display=True):
    '''
    This function performs pose detection on an image.
    Args:
        image: The input image with RGB channels.
        pose: The pose setup function required to perform the pose detection.
        display: A boolean value that is if set to true the function displays the original input image, the resultant image, 
                 and the pose landmarks in 3D plot and returns nothing.
    Returns:
        output_image: The input image with the detected pose landmarks drawn.
        landmarks: A list of detected landmarks converted into their original scale.
    '''
    output_image = image.copy()
    imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(imageRGB)
    height, width, _ = image.shape
    landmarks = []
    
    if results.pose_landmarks:
        if display:
            my_drawing.draw_landmarks(output_image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)  
            
            # Classify the action
            action = classify_action(results.pose_landmarks.landmark)
            cv2.putText(output_image, f'Action: {action}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        
        for landmark in results.pose_landmarks.landmark:
            landmarks.append((int(landmark.x * width), int(landmark.y * height),
                              (landmark.z * width)))
    
    return output_image, landmarks

#%%
# Main Program
# Load video (replace 'running.mp4' with your video file path or use 0 for webcam)
video_path = 0  # Use 0 for webcam or provide a video file path
video = cv2.VideoCapture(video_path)

# Initialize variables
time1 = 0
video.set(3, 1280)  # Set video width
video.set(4, 960)   # Set video height

# Loop through frames
while video.isOpened():
    ok, frame = video.read()
    if not ok:
        print("Failed to read frame from video.")
        break

    frame = cv2.flip(frame, 1)  # Flip frame horizontally
    frame_height, frame_width, _ = frame.shape

    # Resize frame to maintain aspect ratio
    frame = cv2.resize(frame, (int(frame_width * (640 / frame_height)), 640))

    # Pose detection and action classification
    frame, _ = detectPose(frame, pose_video, display=True)

    # Calculate FPS
    time2 = time.time()
    if (time2 - time1) > 0:
        frames_per_second = 1.0 / (time2 - time1)
        cv2.putText(frame, f'FPS: {int(frames_per_second)}', (10, frame.shape[0] - 10), 
                    cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 3)

    # Display frame
    cv2.imshow('Pose Detection', frame)
    time1 = time2

    # Break loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
video.release()
cv2.destroyAllWindows()
