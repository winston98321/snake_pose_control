# -*- coding: utf-8 -*-
"""
Created on Mon Jun 20 18:23:49 2022

@author: PO KAI
"""

"""
Created on Tue Jun 14 21:15:03 2022

@author: PO KAI
"""

# Pose Detections with Model
import cv2
import numpy as np
import mediapipe as mp 
import threading
from new_train import predict
from time import time

def identify(cap,out_video,weight,holistic,mp_drawing,mp_holistic,out):
        ret, frame = cap.read()
        if ret == True:
            # Recolor Feed
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False    

            # Make Detections
            results = holistic.process(image)

            # Recolor image back to BGR for rendering
            image.flags.writeable = True   
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            '''
            mp_drawing.draw_landmarks(
                image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4),
                mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
            )
            '''
            # Pose Detections
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS, 
                                    mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
                                    mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                                    )
            # Export coordinates
            
                # Extract Pose landmarks
            try:
                pose = results.pose_landmarks.landmark
            except:
                return
            pose_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in pose]).flatten())
           
            row =pose_row
            
            # Concate rows
            specify_float = 8
            coords=[]
            for i in range(0,132):
                coords.append(round(row[i], specify_float))
           

            # Concate rows
            body_language_class = predict(weight,np.array(row))
          
            predict_class = body_language_class
            body_language_prob =abs(predict_class-0.5)+0.5
            
            class_probbile = predict_class
            dict_class= 'down' if class_probbile<0.5 else 'up'
          
        
            # Get status box
            cv2.rectangle(image, (0,0), (250, 60), (245, 117, 16), -1)

            # Display Class
            cv2.putText(
                image, 'CLASS', (120,12), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.5, (0, 0, 0), 1, cv2.LINE_AA
            )
            cv2.putText(
                image, dict_class, (120,40), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                1, (255, 255, 255), 2, cv2.LINE_AA
            )

            # Display Probability
            cv2.putText(
                image, 'PROB', (15,12), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.5, (0, 0, 0), 1, cv2.LINE_AA
            )
            body_language_prob = body_language_prob*100
            cv2.putText(
                image, str(round(body_language_prob,2)), 
                (10,40), cv2.FONT_HERSHEY_SIMPLEX, 
                1, (255, 255, 255), 2, cv2.LINE_AA
            )
            out.write(image)
            cv2.imshow('Raw Webcam Feed', image)


        
        else:
            return

def save_display_classify_pose(cap, out_video, weight):
    if (cap.isOpened() == False):
        print("Error opening the video file.")
    else:
        input_fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        output_fps = input_fps - 1
        print(f'Frames per second: {input_fps}')
        print(f'Frame count: {frame_count}')

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f'video_w: {w}, video_h: {h}')

    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # 輸出附檔名為 mp4. 
    out = cv2.VideoWriter(out_video, fourcc, output_fps, (w, h))
    
    mp_drawing = mp.solutions.drawing_utils # Drawing helpers.
    mp_holistic = mp.solutions.holistic     # Mediapipe Solutions.

    # Initiate holistic model
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        start=time()
        while cap.isOpened():
            ret, frame = cap.read()
            if ret == False or (cv2.waitKey(10) & 0xFF == ord('q')):
                break
            t = threading.Thread(target = identify(cap,out_video,weight,holistic,mp_drawing,mp_holistic,out))
            t.start()

    print('Done!')
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(time()-start)

if __name__ == '__main__':

    video_file_name = "test_3"
    
    video_path = "./resource/test/" + video_file_name +".mp4"
    output_video = "./resource/demo/" + video_file_name + "_out_3.mp4"
    #cap = cv2.VideoCapture(0)
    cap = cv2.VideoCapture(video_path)

    b=np.loadtxt('weight.txt')
    save_display_classify_pose(cap=cap, out_video=output_video, weight=(b))
