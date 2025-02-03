import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

# Define action labels
ACTIONS = ['Sitting', 'Standing', 'Running', 'Punching']

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

# Function to extract features from landmarks
def extract_features(landmarks):
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

    # Return feature vector
    return [hip_angle_left, hip_angle_right, elbow_angle_left, elbow_angle_right]

# Train a shallow neural network
# Example: Features = [hip_angle_left, hip_angle_right, elbow_angle_left, elbow_angle_right]
X_train = np.array([[90, 85, 45, 50], [170, 165, 120, 110], [45, 50, 30, 35], [100, 95, 80, 85]])  # Example data
y_train = np.array([0, 1, 2, 3])  # Labels (0: Sitting, 1: Standing, 2: Punching, 3: Running)

# One-hot encode the labels
from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train, num_classes=4)

# Build the neural network model
model = Sequential([
    Dense(16, activation='relu', input_shape=(4,)),  # Input layer (4 features)
    Dense(8, activation='relu'),                    # Hidden layer
    Dense(4, activation='softmax')                  # Output layer (4 classes)
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=50, verbose=0)  # Train for 50 epochs

# Open video capture
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the BGR image to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the image and get the pose landmarks
    results = pose.process(rgb_frame)

    # Check if any pose landmarks are detected
    if results.pose_landmarks:
        # Draw the pose landmarks on the frame
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Extract features and classify the action
        features = extract_features(results.pose_landmarks.landmark)
        prediction = model.predict(np.array([features]))  # Predict action
        action_index = np.argmax(prediction)              # Get the predicted class index
        action = ACTIONS[action_index]                    # Map index to action label
        cv2.putText(frame, f'Action: {action}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Display the frame
    cv2.imshow('Real-Time Action Detection', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
