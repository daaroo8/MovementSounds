import cv2
import mediapipe as mp
import pygame as pg

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

pg.mixer.init()


#TODO: Add the sound files
#TODO: 8 sounds for the hands, 1 for de mouth, 2 for the eyes
sounds =[
  #pg.mixer.Sound("sounds/1.wav"), 
]

def is_finger_down(landmarks, finger_tip, finger_pip):

  if finger_tip == 4:
    return landmarks[finger_tip].x < landmarks[finger_pip].x
  
  else:
    return landmarks[finger_tip].y < landmarks[finger_pip].y

cap = cv2.VideoCapture(0)

with mp_hands.Hands(min_detection_confidence = 0.5, min_tracking_confidence = 0.5,
                    max_num_hands = 2) as hands:
  
  finger_state = [False] * 8
  mouth_state = False
  right_eye_state = False
  left_eye_state = False

  while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
      break

    frame = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
    results = hands.process(frame)

    if results.multi_hand_landmarks:
      for h, hand_landmarks in enumerate(results.multi_hand_landmarks):
        mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS) 

        finger_tips = [4, 8, 12, 16]
        finger_pips = [2, 6, 10, 14] 

        for i in range(4):
          finger_index = i + h*4

          if is_finger_down(hand_landmarks.landmark, finger_tips[i], finger_pips[i]):
            if not finger_state[finger_index]:
              finger_state[finger_index] = True
              #sounds[finger_index].play()
              print(f"Playing sound {finger_index}") 

          else: 
            finger_state[finger_index] = False

    cv2.imshow('Hand Tracking', cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    if cv2.waitKey(1) & 0xFF == 27:
      break

cap.release()
cv2.destroyAllWindows()

