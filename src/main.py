from datetime import datetime as dt
import matplotlib.pyplot as plt
from mediapipe_utils import *
from tqdm import tqdm
import pandas as pd
import click
import glob
import cv2
import os


def load_labels(label_csv_path):
    """Load labels from a CSV file into a Pandas DataFrame."""
    return pd.read_csv(label_csv_path) if label_csv_path else None


def add_label(frame_data, label_df):
    """Add label information to the frame data."""
    if label_df is not None:
        label = label_df.loc[
            label_df["video_name"] == frame_data["video_name"], "label"
        ].values
        frame_data["label"] = label[0] if len(label) > 0 else 0
    else:
        frame_data["label"] = 0


def process_video(
    dataset_name, video_path, output_folder, label_df, landmarks, save_sample_rate
):
    video_name = os.path.basename(video_path)
    cap = cv2.VideoCapture(video_path)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Define DataFrame columns
    df_columns = ["video_name", "frame_count", "label"]
    dynamic_columns = [
        f"{name}_{coord}" for name in HAND_MAPPER.values() for coord in ["x", "y"]
    ]
    dynamic_columns += [
        f"{name}_{coord}" for name in BODY_MAPPER.values() for coord in ["x", "y"]
    ]
    df_columns.extend(dynamic_columns)

    # Initialize empty lists for landmark coordinates
    frame_data = {"video_name": video_name, "frame_count": total_frames}
    for col in dynamic_columns:
        frame_data[col] = []

    add_label(frame_data, label_df)

    sample_output_folder = os.path.join(output_folder, "samples")
    if not os.path.exists(sample_output_folder):
        os.makedirs(sample_output_folder)

    for frame_number in tqdm(range(total_frames), desc=f"Processing {video_name}"):
        ret, frame = cap.read()
        if not ret:
            print(f"Error in frame {frame_number}!")
            break

        hand_landmarks = detect_hand_landmarks(frame) if "hand" in landmarks else {}
        body_landmarks = detect_body_landmarks(frame) if "body" in landmarks else {}

        if frame_number % save_sample_rate == 0:
            sample_file_path = os.path.join(
                sample_output_folder, f"{video_name}_frame_{frame_number}.png"
            )
            cv2.imwrite(sample_file_path, frame)

        for col in dynamic_columns:
            if col in hand_landmarks:
                frame_data[col].extend([hand_landmarks[col]])
            if col in body_landmarks:
                frame_data[col].extend([body_landmarks[col]])

    cap.release()

    current_time = dt.now().strftime("%Y%m%d_%H%M%S")
    output_csv_name = f"{output_folder}/{dataset_name}_{current_time}.csv"

    pd.DataFrame([frame_data], columns=df_columns).to_csv(output_csv_name, index=False)


@click.command()
@click.option("--dataset_name", default="dataset", help="Dataset name.")
@click.option("--folder_path", default="videos/", help="Folder path for videos.")
@click.option(
    "--output_folder", default="annotations/", help="Output folder for annotations."
)
@click.option("--landmarks", default="hand,body", help="List of landmarks.")
@click.option("--label_csv", default=None, help="CSV file for labels.")
@click.option(
    "--save_sample_rate",
    default=100,
    help="Rate at which sample frames will be saved as .png.",
)
def process_videos(
    dataset_name, folder_path, output_folder, landmarks, label_csv, save_sample_rate
):
    """Main function to process all video files."""
    video_files = glob.glob(f"{folder_path}/**/*.mp4", recursive=True) + glob.glob(
        f"{folder_path}/**/*.avi", recursive=True
    )
    label_df = load_labels(label_csv)
    for video_path in video_files:
        process_video(
            dataset_name,
            video_path,
            output_folder,
            label_df,
            landmarks.split(","),
            save_sample_rate,
        )


if __name__ == "__main__":
    process_videos()
