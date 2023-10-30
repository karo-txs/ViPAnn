from mediapipe_utils import (
    face_landmarks_from_frame,
    hand_landmarks_from_frame,
    pose_landmarks_from_frame,
)
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import pandas as pd
import click
import glob
import cv2
import os


def process_video(video_path, participant_id, sequence_id, results_path):
    """Extract landmarks from a video and save to parquet format."""

    if not os.path.exists(video_path):
        with open(os.path.join(results_path, "missing.txt"), "a") as f:
            f.write(f"{video_path}\n")
        return None

    cap = cv2.VideoCapture(video_path)
    frames_data = []
    frame_num = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frames_data.extend(hand_landmarks_from_frame(frame, frame_num))
        frames_data.extend(pose_landmarks_from_frame(frame, frame_num))
        frames_data.extend(face_landmarks_from_frame(frame, frame_num))

        frame_num += 1

    cap.release()
    output_path = os.path.join(
        results_path, f"landmark_files/{participant_id}/{sequence_id}.parquet"
    )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    pd.DataFrame(frames_data).to_parquet(output_path)
    return output_path


@click.command()
@click.option(
    "--base_file", help="The base CSV file containing video names and labels."
)
@click.option("--video_path", help="Path to the folder containing videos.")
@click.option(
    "--results_path", default="./results", help="Path where the results will be saved."
)
@click.option("--workers", default=4, help="Number of workers for parallel processing.")
def main(base_file, video_path, results_path, workers):
    """Extract landmarks from videos and generate train.csv"""
    base_df = pd.read_csv(base_file)
    os.makedirs(os.path.dirname(results_path), exist_ok=True)

    labels = base_df["label"].tolist()

    video_paths = [os.path.join(video_path, v) for v in base_df["video_name"].tolist()]
    existing_video_indices = [
        i for i, path in enumerate(video_paths) if os.path.exists(path)
    ]
    video_paths = [video_paths[i] for i in existing_video_indices]

    labels = [labels[i] for i in existing_video_indices]

    # Convert labels to numeric and save the mapping
    unique_labels = sorted(list(set(labels)))
    label_to_num = {label: i for i, label in enumerate(unique_labels)}
    num_labels = [label_to_num[label] for label in labels]
    pd.DataFrame(list(label_to_num.items()), columns=["sign", "number"]).to_csv(
        os.path.join(results_path, "sign_mapping.csv"), index=False
    )

    if "participant_id" in base_df.columns:
        participant_ids = [
            base_df["participant_id"].tolist()[i] for i in existing_video_indices
        ]
    else:
        participant_ids = [0] * len(video_paths)

    sequence_ids = list(range(1001, 1001 + len(video_paths)))

    video_data = list(
        zip(
            video_paths,
            participant_ids,
            sequence_ids,
            [results_path] * len(video_paths),
        )
    )
    with ProcessPoolExecutor(max_workers=workers) as executor:
        output_paths = list(
            tqdm(executor.map(process_video, *zip(*video_data)), total=len(video_data))
        )

    output_paths = glob.glob(
        f"{results_path}/landmark_files/*/*.parquet", recursive=True
    )

    train_df = pd.DataFrame(
        {
            "path": [output.split("/")[-1] for output in output_paths],
            "participant_id": participant_ids,
            "sequence_id": sequence_ids,
            "sign": num_labels,
        }
    )
    train_df.to_csv(os.path.join(results_path, "landmarks.csv"), index=False)


if __name__ == "__main__":
    main()
