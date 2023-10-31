import mediapipe as mp
import cv2


mp_hands = mp.solutions.hands
mp_face = mp.solutions.face_mesh
mp_pose = mp.solutions.pose


def validate_landmark(landmark_data):
    """Ensure the landmark values are within the expected range."""
    # Ensure x and y are within [0, 1]
    landmark_data["x"] = min(max(landmark_data["x"], 0), 1)
    landmark_data["y"] = min(max(landmark_data["y"], 0), 1)
    # For z, we simply set it to 0 if it's out of range
    if not (
        -1 <= landmark_data["z"] <= 1
    ):  # this range is just an example, you can adjust it based on your needs
        landmark_data["z"] = 0
    return landmark_data


def hand_landmarks_from_frame(image, frame_num):
    frames_data = []
    with mp_hands.Hands(
        static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5
    ) as hands:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)

        if results.multi_hand_landmarks:
            for hand_index, hand_landmark in enumerate(results.multi_hand_landmarks):
                # Determine hand type using multi_handedness
                if results.multi_handedness[hand_index].classification[0].label == "Left":
                    landmark_type = "left_hand"
                else:
                    landmark_type = "right_hand"

                for index, landmark in enumerate(hand_landmark.landmark):
                    frames_data.append(
                        validate_landmark(
                            {
                                "frame": frame_num,
                                "row_id": f"{frame_num}-{landmark_type}-{index}",
                                "type": landmark_type,
                                "landmark_index": index,
                                "x": landmark.x,
                                "y": landmark.y,
                                "z": landmark.z,
                            }
                        )
                    )
    return frames_data


def pose_landmarks_from_frame(image, frame_num):
    frames_data = []
    landmark_type = "pose"
    with mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5) as pose:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)

        if results.pose_landmarks:
            for index, landmark in enumerate(results.pose_landmarks.landmark):
                frames_data.append(
                    validate_landmark(
                        {
                            "frame": frame_num,
                            "row_id": f"{frame_num}-{landmark_type}-{index}",
                            "type": landmark_type,
                            "landmark_index": index,
                            "x": landmark.x,
                            "y": landmark.y,
                            "z": landmark.z,
                        }
                    )
                )

    return frames_data


def face_landmarks_from_frame(image, frame_num):
    frames_data = []
    landmark_type = "face"
    with mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True, min_detection_confidence=0.5
    ) as face_mesh:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(image_rgb)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                for index, landmark in enumerate(face_landmarks.landmark):
                    frames_data.append(
                        validate_landmark(
                            {
                                "frame": frame_num,
                                "row_id": f"{frame_num}-{landmark_type}-{index}",
                                "type": landmark_type,
                                "landmark_index": index,
                                "x": landmark.x,
                                "y": landmark.y,
                                "z": landmark.z,
                            }
                        )
                    )

    return frames_data
