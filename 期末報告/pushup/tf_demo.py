"""
Created on Tue Jun 14 21:15:03 2022

@author: PO KAI
"""

# Pose Detections with Model
import cv2
import numpy as np
import pandas as pd
import mediapipe as mp 
import pickle
import tensorflow as tf



def save_display_classify_pose(cap, model, out_video):
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
        
        while cap.isOpened():
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
                pose = results.pose_landmarks.landmark
                pose_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in pose]).flatten())
               
                row =pose_row
                
                # Concate rows
                specify_float = 8
                coords=[]
                for i in range(0,132):
                    coords.append(round(row[i], specify_float))
               
    
                outdic =np.expand_dims(np.array(coords, dtype=np.float32), axis=0)
                
           
                # Concate rows
                body_language_class = model.predict(outdic)
                #body_language_prob = model.predict_proba(input_dict)[0]
                # print(f'class: {body_language_class}, prob: {body_language_prob}')
                predict_class = body_language_class
                body_language_prob =(max(predict_class[0]))
                
                class_probbile = predict_class[0].tolist().index(body_language_prob)
                dict_class={
                    1:'down',
                    0:'up'
                    }
                print(f'\n prob: {body_language_prob},class: {dict_class[class_probbile]}\n')
                print(f'\n prob: {body_language_prob},class: {dict_class[class_probbile]}\n')
               

                # Grab ear coords
               

                # Get status box
                cv2.rectangle(image, (0,0), (250, 60), (245, 117, 16), -1)

                # Display Class
                cv2.putText(
                    image, 'CLASS', (120,12), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, (0, 0, 0), 1, cv2.LINE_AA
                )
                cv2.putText(
                    image, dict_class[class_probbile], (120,40), 
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

                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break
            
            else:
                break

    print('Done!')
    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':

    # Test video file name: cat_camel2, bridge2, heel_raise2.
    video_file_name = "test_3"
    #model_weights = './model_weights/weights_test.pkl'
    model_weights = './model_weights/pushup-classifier.h5'
    
    video_path = "./resource/test/" + video_file_name +".mp4"
    output_video = "./resource/demo/" + video_file_name + "_out_2.mp4"

    cap = cv2.VideoCapture(video_path)

    # Load Model.
    model = tf.keras.models.load_model(model_weights)
    '''with open(model_weights, 'rb') as f:
        model = pickle.load(f)'''
        
    
    
    # display_classify_pose(cap=cap, model=model)
    save_display_classify_pose(cap=cap, model=model, out_video=output_video)