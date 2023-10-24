from concurrent.futures import ProcessPoolExecutor
from mediapipe.mediapipe_utils import *
from datetime import datetime as dt
from multiprocessing import Manager
from threading import Thread
from tqdm import tqdm
import pandas as pd
import click
import time
import glob
import cv2
import os


def load_labels(label_csv_path):
    return pd.read_csv(label_csv_path) if label_csv_path else None


def add_label(frame_data, label_df):
    if label_df is not None:
        label = label_df.loc[
            label_df["video_name"] == frame_data["video_name"], "label"
        ].values
        frame_data["label"] = label[0] if len(label) > 0 else 0
    else:
        frame_data["label"] = 0


def create_dataframe_columns():
    df_columns = [
        "video_name",
        "frame_count",
        "label",
        "video_size_height",
        "video_size_width",
        "video_fps",
    ]
    dynamic_columns = [
        f"{name}_{'left' if direction == 0 else 'right'}_{coord}"
        for name in HAND_MAPPER.values()
        for direction in [0, 1]
        for coord in ["x", "y"]
    ]
    dynamic_columns += [
        f"{name}_{coord}" for name in BODY_MAPPER.values() for coord in ["x", "y", "z"]
    ]
    df_columns.extend(dynamic_columns)
    return dynamic_columns, df_columns


def process_video(
    output_queue,
    video_path,
    label_df,
    landmarks,
):
    video_name = os.path.basename(video_path)
    cap = cv2.VideoCapture(video_path)
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    dynamic_columns, df_columns = create_dataframe_columns()

    frame_data = {
        "video_name": video_name,
        "frame_count": total_frames,
        "label": 0,
        "video_size_height": video_height,
        "video_size_width": video_width,
        "video_fps": video_fps,
    }
    for col in dynamic_columns:
        frame_data[col] = []

    add_label(frame_data, label_df)

    for frame_number in range(total_frames):
        ret, frame = cap.read()

        if not ret:
            print(f"Error in frame {frame_number}!")
            break

        hand_landmarks = detect_hand_landmarks(frame) if "hand" in landmarks else {}
        body_landmarks = detect_body_landmarks(frame) if "body" in landmarks else {}

        for hand, marks in hand_landmarks.items():
            for col_name, col_value in marks.items():
                coord = col_name.split("_")[-1] 
                landmark_name = "_".join(col_name.split("_")[:-1])

                full_col_name = f"{landmark_name}_{hand}_{coord}"

                if full_col_name in frame_data:
                    frame_data[full_col_name].extend([col_value])

            if not marks:
                for col_name in HAND_MAPPER.values():
                    for coord in ["x", "y"]:
                        full_col_name = f"{col_name}_{hand}_{coord}"
                        if full_col_name in frame_data:
                            frame_data[full_col_name].extend([0])

        for col in dynamic_columns:
            if col in body_landmarks:
                frame_data[col].extend([body_landmarks[col]])

    cap.release()

    df = pd.DataFrame([frame_data], columns=df_columns)
    output_queue.put(df)


def concatenate_pkls(output_folder, final_output_name):
    print(f"\n> Saving final result in {final_output_name}")

    all_files = glob.glob(os.path.join(output_folder, "*.pkl"))
    _, df_columns = create_dataframe_columns()
    all_data = pd.DataFrame(columns=df_columns)

    for file in all_files:
        df = pd.read_pickle(file)
        all_data = pd.concat([all_data, df], ignore_index=True)

    all_data.to_pickle(final_output_name)


def process_output_queue(output_queue, save_step, output_folder, dataset_name):
    all_data = pd.DataFrame()
    num_rows = 0

    while True:
        if not output_queue.empty():
            df = output_queue.get()
            all_data = pd.concat([all_data, df], ignore_index=True)
            num_rows += len(df)

            if num_rows >= save_step:
                current_time = dt.now().strftime("%Y%m%d_%H%M%S")
                output_csv_name = f"{output_folder}/{dataset_name}_{current_time}.pkl"
                print(f"\n> Saving partial result in {output_csv_name}")
                all_data.to_pickle(output_csv_name)
                all_data = pd.DataFrame()
                num_rows = 0
        else:
            time.sleep(1)


@click.command()
@click.option("--dataset_name", default="dataset", help="Dataset name.")
@click.option("--folder_path", default="videos", help="Folder path for videos.")
@click.option(
    "--output_folder", default="annotations", help="Output folder for annotations."
)
@click.option("--landmarks", default="hand,body", help="List of landmarks.")
@click.option("--label_csv", default=None, help="CSV file for labels.")
@click.option("--workers", default=5, help="Number of workers.")
@click.option(
    "--max_num_samples",
    default=None,
    type=int,
    help="Maximum number of videos to process.",
)
@click.option(
    "--save_step",
    default=500,
    type=int,
    help="Step to partial save.",
)
def process_videos(
    dataset_name,
    folder_path,
    output_folder,
    landmarks,
    label_csv,
    workers,
    max_num_samples,
    save_step,
):
    video_files = glob.glob(f"{folder_path}/**/*.mp4", recursive=True) + glob.glob(
        f"{folder_path}/**/*.avi", recursive=True
    )

    if max_num_samples is not None:
        video_files = video_files[:max_num_samples]

    label_df = load_labels(label_csv)

    manager = Manager()
    output_queue = manager.Queue()

    # Save dataframe
    process_thread = Thread(
        target=process_output_queue,
        args=(output_queue, save_step, output_folder, dataset_name),
    )
    process_thread.start()

    _, df_columns = create_dataframe_columns()

    all_data = pd.DataFrame(columns=df_columns)

    current_time = dt.now().strftime("%Y%m%d_%H%M%S")
    output_csv_name = f"{output_folder}/{dataset_name}_{current_time}.pkl"

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(
                process_video,
                output_queue,
                video_path,
                label_df,
                landmarks.split(","),
            )
            for video_path in video_files
        ]

        for idx, future in tqdm(
            enumerate(futures), total=len(futures), desc="Processing Videos"
        ):
            future.result()

        if not all_data.empty:
            current_time = dt.now().strftime("%Y%m%d_%H%M%S")
            output_csv_name = (
                f"{output_folder}/{dataset_name}_{current_time}_batch_last.pkl"
            )
            all_data.to_pickle(output_csv_name)

    concatenate_pkls("annotations/", "annotations/final_dataset.pkl")
    print("Process completed successfully.")


if __name__ == "__main__":
    process_videos()
