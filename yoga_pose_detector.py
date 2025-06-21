import os
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
mp_drawing = mp.solutions.drawing_utils #gives all our drawing utilities, visualise our poses
mp_pose = mp.solutions.pose #importing pose estimation models
pose = mp_pose.Pose(static_image_mode=True)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib
import pyttsx3  # Text-to-speech engine
import time


data = pd.read_csv('yoga_pose_angles_dataset.csv')


def calculate_angle(a,b,c):
    a=np.array(a)#First
    b=np.array(b)#Second
    c=np.array(c)#Third
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle > 180.0:
        angle = 360.0-angle
        
    return angle
                

def run_yoga_pose_detection():
 
    clf = joblib.load("pose_classifier.pkl")
    le = joblib.load("label_encoder.pkl")
   
    # Compute median joint angles per pose for less strict correction
    median_angles_per_pose = data.groupby('pose').median()

    # Text-to-speech setup
    engine = pyttsx3.init()
    def speak(text):
        engine.say(text)
        engine.runAndWait()

    cap = cv2.VideoCapture(0)
    prev_label = ""
    spoken_joints = set()
    detection_time = None
    pose_confirmed = False

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = pose.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            label = "No yoga pose detected"
            landmark_colors = {}

            try:
                landmarks = results.pose_landmarks.landmark
                def get_coords(idx):
                    lm = landmarks[idx]
                    return [lm.x, lm.y]

                # Compute joint angles
                left_elbow = calculate_angle(get_coords(11), get_coords(13), get_coords(15))
                right_elbow = calculate_angle(get_coords(12), get_coords(14), get_coords(16))
                left_shoulder = calculate_angle(get_coords(13), get_coords(11), get_coords(23))
                right_shoulder = calculate_angle(get_coords(14), get_coords(12), get_coords(24))
                left_knee = calculate_angle(get_coords(23), get_coords(25), get_coords(27))
                right_knee = calculate_angle(get_coords(24), get_coords(26), get_coords(28))
                left_hip = calculate_angle(get_coords(11), get_coords(23), get_coords(25))
                right_hip = calculate_angle(get_coords(12), get_coords(24), get_coords(26))
                left_ankle = calculate_angle(get_coords(25), get_coords(27), get_coords(31))
                right_ankle = calculate_angle(get_coords(26), get_coords(28), get_coords(32))
                left_wrist = calculate_angle(get_coords(19), get_coords(15), get_coords(13))
                right_wrist = calculate_angle(get_coords(20), get_coords(16), get_coords(14))
                spine_angle = calculate_angle(get_coords(23), get_coords(11), get_coords(12))
                nose_x = landmarks[0].x
                nose_y = landmarks[0].y

                angles = [left_elbow, right_elbow, left_shoulder, right_shoulder,
                        left_knee, right_knee, left_hip, right_hip,
                        left_ankle, right_ankle, left_wrist, right_wrist,
                        spine_angle, nose_x, nose_y]

                feature_cols = ['left_elbow', 'right_elbow', 'left_shoulder', 'right_shoulder',
                                'left_knee', 'right_knee', 'left_hip', 'right_hip',
                                'left_ankle', 'right_ankle', 'left_wrist', 'right_wrist',
                                'spine_angle', 'nose_x', 'nose_y']

                X_live = pd.DataFrame([angles], columns=feature_cols)
                proba = clf.predict_proba(X_live)[0]
                max_conf = np.max(proba)
                pred = clf.predict(X_live)[0]

                if max_conf < 0.7:
                    label = "No yoga pose detected"
                    pose_confirmed = False
                    detection_time = None
                    spoken_joints.clear()
                else:
                    label = le.inverse_transform([pred])[0]
                    if label != prev_label:
                        prev_label = label
                        detection_time = time.time()
                        pose_confirmed = False
                        spoken_joints.clear()
                        speak(f"Pose detected: {label}. Hold the pose.")
                    elif not pose_confirmed and time.time() - detection_time >= 5:
                        pose_confirmed = True

                if label in median_angles_per_pose.index:
                    median_pose_angles = median_angles_per_pose.loc[label]

                    joint_angles = {
                        "left_elbow": (left_elbow, get_coords(13)),
                        "right_elbow": (right_elbow, get_coords(14)),
                        "left_knee": (left_knee, get_coords(25)),
                        "right_knee": (right_knee, get_coords(26)),
                        "left_shoulder": (left_shoulder, get_coords(11)),
                        "right_shoulder": (right_shoulder, get_coords(12)),
                        "left_hip": (left_hip, get_coords(23)),
                        "right_hip": (right_hip, get_coords(24)),
                        "left_ankle": (left_ankle, get_coords(27)),
                        "right_ankle": (right_ankle, get_coords(28)),
                        "left_wrist": (left_wrist, get_coords(15)),
                        "right_wrist": (right_wrist, get_coords(16))
                    }

                    for joint, (angle, coords) in joint_angles.items():
                        if joint in median_pose_angles:
                            ref = median_pose_angles[joint]
                            deviation = abs(angle - ref)
                            if deviation <= 20:
                                color = (0, 255, 0)
                            else:
                                color = (0, 0, 255)
                                if pose_confirmed and joint not in spoken_joints:
                                    speak(f"Please adjust your {joint.replace('_', ' ')}")
                                    spoken_joints.add(joint)
                        else:
                            color = (255, 255, 255)

                        landmark_colors[joint] = color
                        coords = tuple(np.multiply(coords, [image.shape[1], image.shape[0]]).astype(int))
                        cv2.putText(image, f"{round(angle,1)}", coords,
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)
                        
                    if pose_confirmed and all(
                        joint in median_pose_angles and abs(angle - median_pose_angles[joint]) <= 20
                        for joint, (angle, _) in joint_angles.items()
                    ):
                        if "all_clear_spoken" not in spoken_joints:
                            speak("Bravo you nailed it")
                            spoken_joints.add("all_clear_spoken")



            except Exception as e:
                print(f"[ERROR] {e}")
                

            # Draw landmarks with corrected color overlays
            if results.pose_landmarks:
                for connection in mp_pose.POSE_CONNECTIONS:
                    start_idx, end_idx = connection
                    start_point = landmarks[start_idx]
                    end_point = landmarks[end_idx]
                    start = (int(start_point.x * image.shape[1]), int(start_point.y * image.shape[0]))
                    end = (int(end_point.x * image.shape[1]), int(end_point.y * image.shape[0]))

                    color = (200, 200, 200)  # default gray
                    for joint, color_val in landmark_colors.items():
                        joint_idx = mp_pose.PoseLandmark[joint.upper()].value if hasattr(mp_pose.PoseLandmark, joint.upper()) else None
                        if joint_idx in connection:
                            color = color_val
                            break

                    cv2.line(image, start, end, color, 2)

            cv2.putText(image, f"Pose: {label}", (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)

            cv2.imshow("Yoga Pose Detection", image)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_yoga_pose_detection()