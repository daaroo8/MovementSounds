import cv2
import mediapipe as mp
import pygame as pg
import math
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Esto suprime los mensajes de información y advertencias

import tensorflow as tf

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh


pg.mixer.init()

#TIP: FIRST YOU SHOULD SHOW TO THE CAM THE RIGTH HAND
sounds =[
  pg.mixer.Sound("fa.wav"), 
  pg.mixer.Sound("mi.wav"), 
  pg.mixer.Sound("re.wav"), 
  pg.mixer.Sound("doG.wav"), 
  pg.mixer.Sound("sol.wav"), 
  pg.mixer.Sound("la.wav"), 
  pg.mixer.Sound("si.mp3"), 
  pg.mixer.Sound("doA.wav")
]


sounds_print = [
"fa",
"mi",
"re",
"doG",
"sol",
"la",
"si",
"doA"
]

def is_finger_down(landmarks, finger_tip, finger_pip):

  if finger_tip == 4:
    return landmarks[finger_tip].x < landmarks[finger_pip].x
  
  else:
    return landmarks[finger_tip].y < landmarks[finger_pip].y
  
def euclidean_distance(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

def mouth_aspect_ratio(landmarks, img_width, img_height):
    # Accede a los puntos de los labios por su índice en la lista de landmarks
    top_lip = (int(landmarks.landmark[13].x * img_width), int(landmarks.landmark[13].y * img_height))
    bottom_lip = (int(landmarks.landmark[14].x * img_width), int(landmarks.landmark[14].y * img_height))
    left_mouth = (int(landmarks.landmark[78].x * img_width), int(landmarks.landmark[78].y * img_height))
    right_mouth = (int(landmarks.landmark[308].x * img_width), int(landmarks.landmark[308].y * img_height))
    
    # Calcular la distancia vertical y horizontal
    vertical = euclidean_distance(top_lip, bottom_lip)
    horizontal = euclidean_distance(left_mouth, right_mouth)
    
    return vertical / horizontal if horizontal else 0

def is_mouth_open(landmarks):
  return landmarks[10].y < landmarks[8].y

MOUTH_THRESHOLD = 0.4

cap = cv2.VideoCapture(0)

with mp_hands.Hands(min_detection_confidence = 0.5, min_tracking_confidence = 0.5,
                    max_num_hands = 2) as hands:
  with mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, 
                             min_detection_confidence=0.5) as face_mesh:

    drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
    finger_state = [False] * 8

    while cap.isOpened():
      ret, frame = cap.read()
      if not ret:
        break

      frame = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
      results_hands = hands.process(frame)
      results_face = face_mesh.process(frame)
      img_h, img_w = frame.shape[:2]
      if results_face.multi_face_landmarks:
        face_landmarks = results_face.multi_face_landmarks[0]
        mp_drawing.draw_landmarks(frame, face_landmarks, mp_face_mesh.FACEMESH_CONTOURS, drawing_spec, drawing_spec)


        if results_hands.multi_hand_landmarks and results_face.multi_face_landmarks:
          for h, hand_landmarks in enumerate(results_hands.multi_hand_landmarks):
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS) 
            mp_drawing.draw_landmarks(frame, face_landmarks, mp_face_mesh.FACEMESH_CONTOURS, drawing_spec, drawing_spec)
            
            finger_tips = [4, 8, 12, 16]
            finger_pips = [2, 6, 10, 14] 

            for i in range(4):
              finger_index = i + h * 4

            #If the mouth is open, it simulate th use of the pedal
            # if mouth_aspect_ratio(face_landmarks, img_w, img_h) > MOUTH_THRESHOLD:
            #   print("Pedal pressed")

            #   if is_finger_down(hand_landmarks.landmark, finger_tips[i], finger_pips[i]):
                  
            #     finger_state[finger_index] = True
            #     sounds[finger_index].play()
                
            #     print(f"Playing sound {sounds_print[finger_index]}") 
            #     print(f"Playing sound {finger_index}")

            # else:
              if is_finger_down(hand_landmarks.landmark, finger_tips[i], finger_pips[i]):

                if not finger_state[finger_index]:
                  finger_state[finger_index] = True
                  sounds[finger_index].play()
                  print(f"Playing sound {sounds_print[finger_index]}") 
                  print(f"Playing sound {finger_index}")

              else: 
                finger_state[finger_index] = False

        cv2.imshow('Hand Tracking', cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

        if cv2.waitKey(1) & 0xFF == 27:
          break

cap.release()
cv2.destroyAllWindows()

