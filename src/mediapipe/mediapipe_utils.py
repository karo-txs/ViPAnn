from mediapipe_connections import HAND_MAPPER, BODY_MAPPER
import mediapipe as mp
import cv2


mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
mp_face = mp.solutions.face_mesh
mp_pose = mp.solutions.pose


def detect_hand_landmarks(image):
    mp_hands = mp.solutions.hands
    hand_landmarks = {"left": {}, "right": {}}
    with mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=2,
        min_detection_confidence=0.35,
        min_tracking_confidence=0.35,
    ) as hands:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)

        if results.multi_hand_landmarks:
            for hand_index, hand_landmark in enumerate(results.multi_hand_landmarks):
                handedness = (
                    results.multi_handedness[hand_index].classification[0].label.lower()
                )
                mp_drawing.draw_landmarks(
                    image, hand_landmark, mp_hands.HAND_CONNECTIONS
                )

                for id, lm in enumerate(hand_landmark.landmark):
                    landmark_name = HAND_MAPPER.get(id, str(id))
                    hand_landmarks[handedness][f"{landmark_name}_x"] = lm.x
                    hand_landmarks[handedness][f"{landmark_name}_y"] = lm.y
    return hand_landmarks


def detect_body_landmarks(image):
    mp_pose = mp.solutions.pose
    body_landmarks = {}
    with mp_pose.Pose(
        static_image_mode=True,
        min_detection_confidence=0.35,
        min_tracking_confidence=0.35,
    ) as pose:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)

        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                image, results.pose_landmarks, mp_hands.HAND_CONNECTIONS
            )
            for id, lm in enumerate(results.pose_landmarks.landmark):
                landmark_name = BODY_MAPPER.get(id, str(id))
                body_landmarks[f"{landmark_name}_x"] = lm.x
                body_landmarks[f"{landmark_name}_y"] = lm.y

    return body_landmarks
