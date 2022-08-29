import mediapipe as mp
import cv2               #for web cam feed
import numpy
import numpy as np       #work with outputs
import uuid              #uniform unique identifier
import os                #operating system





#Using Mediapipe Hands

mp_drawing = mp.solutions.drawing_utils  #use t draw all the landmarks to the screen
mp_hands = mp.solutions.hands            #hands module

# os.mkdir('Output Images')
#working with webcam

cap = cv2.VideoCapture(0)    #getting web cam feed

with mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5) as hands:  #instantiating mediapipe hands model..initially detecting hand and then tracking

    while cap.isOpened():          #while connected to web cam
            ret, frame = cap.read()   #Read each frame from our webcam //frame=image from web cam

            #Color
            image =cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)  #converting to RGB
            #set flag
            image.flags.writeable = False
            #detections
            results = hands.process(image)
            canvas =numpy.zeros_like(image)
            image.flags.writeable = True
            #recolor
            image = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
            print(results)

            if results.multi_hand_landmarks:
                for num, hand in enumerate(results.multi_hand_landmarks):
                    mp_drawing.draw_landmarks(canvas, hand, mp_hands.HAND_CONNECTIONS,
                                              mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),#drawing specs
                                              mp_drawing.DrawingSpec(color=(250, 44, 250), thickness=2,
                                                                      circle_radius=2),
                                              )

            cv2.imwrite(os.path.join('Output Images', '{}.jpg'.format(uuid.uuid1())), image)#saving


            cv2.imshow("Hand Tracking",canvas) #passes frame name and frame

            if cv2.waitKey(10) & 0xFF == ord('q'): #to stop the feed
                break




cap.release()   #release webcam
cv2.destroyAllWindows()  #closess down the frame


