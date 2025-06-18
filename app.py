import streamlit as st
import cv2
import numpy as np
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Initialize MediaPipe drawing and pose
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

def calculate_angle(a,b,c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = np.arctan2(c[1] - b[1],c[0] - b[0]) - np.arctan2(a[1]-b[1],a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)

    if angle > 180.0:
        angle = 360 - angle
    return angle

def squats(landmarks, image, blank_image, counter, state):
    left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                     landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
    right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                      landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]

    left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                  landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
    right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                   landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]

    left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                  landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
    right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                   landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]

    L_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
    R_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)

    # Display angles
    cv2.putText(blank_image, f'{int(L_angle)}', tuple(np.multiply(left_elbow, [640, 480]).astype(int)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    cv2.putText(blank_image, f'{int(R_angle)}', tuple(np.multiply(right_elbow, [640, 480]).astype(int)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # Draw background box
    cv2.rectangle(blank_image, (50, 50), (300, 150), (255, 0, 0), -1)

    # Curl logic
    if L_angle > 160 and R_angle > 160:
        state = "down"
    if L_angle < 50 and R_angle < 50 and state == "down":
        state = "up"
        counter += 1

    # Display Reps
    cv2.putText(blank_image, f'Reps: {str(counter)}', (70, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)

    return counter, state

# Streamlit UI setup
st.set_page_config(layout="wide")
st.title("💪 Bicep Curl Counter")

# Start webcam checkbox
run = st.checkbox('📸 Start Webcam')

# Layout: two columns
col1, col2 = st.columns(2)
with col1:
    st.subheader("Live Webcam Feed")
    live_feed = st.image([], channels="RGB")

with col2:
    st.subheader("Pose Estimation")
    live_output = st.image([], channels="RGB")

# Initialize variables
counter = 0
state = None

# Main loop
if run:
    cap = cv2.VideoCapture(0)

    with mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                st.warning("⚠️ Failed to capture frame.")
                break

            frame = cv2.resize(frame, (640, 480))
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            blank_image = np.zeros_like(image)

            image.flags.writeable = False
            results = pose.process(image)
            image.flags.writeable = True

            if results.pose_landmarks:
                try:
                    landmarks = results.pose_landmarks.landmark
                    counter, state = squats(landmarks, image, blank_image, counter, state)
                    mp_drawing.draw_landmarks(
                        blank_image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                except Exception as e:
                    st.error(f"Error in squats function: {e}")

            # Display on UI
            live_feed.image(image, channels="RGB")
            live_output.image(blank_image, channels="RGB")
            st.markdown(f"### ✅ Reps Count: `{counter}`")

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
