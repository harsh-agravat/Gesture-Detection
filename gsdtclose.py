import cv2
import mediapipe as mp
import numpy as np
import pyautogui

# Initialize MediaPipe Hands and OpenCV
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.7)  # Detect up to 2 hands
cap = cv2.VideoCapture(0)

def classify_gesture(hand_landmarks):
    finger_tips = [
        hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP],
        hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP],
        hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP],
        hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
    ]
    
    is_fist = all(tip.y > hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].y for tip in finger_tips)
    
    if is_fist:
        return "Fist"
    else:
        return "Open Hand"

def perform_action(gesture):
    if gesture == "Fist":
        pyautogui.press('volumedown')
    elif gesture == "Open Hand":
        pyautogui.press('volumeup')

# To track if both hands closed then opened
both_fists = False

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    # List to store gestures for both hands
    gestures = []

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            gesture = classify_gesture(hand_landmarks)
            gestures.append(gesture)  # Track the gestures for both hands
            perform_action(gesture)
            cv2.putText(frame, gesture, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Check if both hands are making fists
        if len(gestures) == 2 and gestures == ["Fist", "Fist"]:
            both_fists = True

        # Check if both hands open after making fists
        if both_fists and len(gestures) == 2 and gestures == ["Open Hand", "Open Hand"]:
            print("Both hands opened after being fists. Exiting...")
            break  # Exit the loop when both hands open after being fists

    # Display the frame
    cv2.imshow('Gesture Recognition', frame)

    # Optionally, press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
