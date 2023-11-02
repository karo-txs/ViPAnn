import mediapipe as mp
import math
import cv2

mp_holistic = mp.solutions.holistic

ROWS_PER_FRAME = 543
MAX_LEN = 384
CROP_LEN = MAX_LEN
PAD = math.nan 

def holistic_landmarks_from_frame(image, frame_num):
    # Inicialize o frames_data com valores PAD para todos os índices esperados
    frames_data = [{'frame': frame_num,
                    'row_id': f"{frame_num}-pad-{index}",
                    'type': 'pad',
                    'landmark_index': index,
                    'x': PAD,
                    'y': PAD,
                    'z': PAD} for index in range(ROWS_PER_FRAME)]

    with mp_holistic.Holistic(
        static_image_mode=True, 
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as holistic:
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = holistic.process(image_rgb)

        # Extract face landmarks
        if results.face_landmarks:
            for index, landmark in enumerate(results.face_landmarks.landmark):
                frames_data[index].update({
                    "row_id": f"{frame_num}-face-{index}",
                    "type": "face",
                    "x": landmark.x,
                    "y": landmark.y,
                    "z": landmark.z,
                })
        
        # Extract left hand landmarks
        if results.left_hand_landmarks:
            for index, landmark in enumerate(results.left_hand_landmarks.landmark):
                frames_data[468 + index].update({
                    "row_id": f"{frame_num}-left_hand-{index}",
                    "type": "left_hand",
                    "x": landmark.x,
                    "y": landmark.y,
                    "z": landmark.z,
                })

        # Extract pose landmarks
        if results.pose_landmarks:
            for index, landmark in enumerate(results.pose_landmarks.landmark):
                frames_data[489 + index].update({
                    "row_id": f"{frame_num}-pose-{index}",
                    "type": "pose",
                    "x": landmark.x,
                    "y": landmark.y,
                    "z": landmark.z,
                })

        # Extract right hand landmarks
        if results.right_hand_landmarks:
            for index, landmark in enumerate(results.right_hand_landmarks.landmark):
                frames_data[522 + index].update({
                    "row_id": f"{frame_num}-rigth_hand-{index}",
                    "type": "rigth_hand",
                    "x": landmark.x,
                    "y": landmark.y,
                    "z": landmark.z,
                })

    return frames_data