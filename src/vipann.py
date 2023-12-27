from mediapipe_utils import holistic_landmarks_from_frame
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import pandas as pd
import click
import errno
import glob
import cv2
import os


def mkdir(path):
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def find_file(search_path, target_file):
    for dirpath, dirnames, filenames in os.walk(search_path):
        if target_file in filenames:
            return os.path.join(dirpath, target_file)
    return None


def process_video(video_path, results_path):
    """Extract landmarks from a video and save to parquet format."""

    if not os.path.exists(video_path):
        with open(os.path.join(results_path, "missing.txt"), "a") as f:
            f.write(f"{video_path}\n")
        return None

    video_path = video_path.replace("\\", "/")
    name = video_path.split("/")[-1].replace(".mp4", "")

    cap = cv2.VideoCapture(video_path)
    frames_data = []
    frame_num = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frames_data.extend(holistic_landmarks_from_frame(frame, frame_num))

        frame_num += 1

    cap.release()
    output_path = os.path.join(results_path, f"landmark_files/{name}.parquet")
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
@click.option("--workers", default=1, help="Number of workers for parallel processing.")
@click.option("--class_map", default=None, help="Path to the class map file.")
def main(base_file, video_path, results_path, workers, class_map):
    """Extract landmarks from videos and generate train.csv"""
    base_df = pd.read_csv(base_file)
    mkdir(results_path)

    labels = base_df["sign"].tolist()
    video_paths = [find_file(video_path, v) for v in base_df["path"].tolist()]
    existing_video_indices = [idx for idx, v in enumerate(video_paths) if v]
    video_paths = [video_paths[i] for i in existing_video_indices]
    labels = [labels[i] for i in existing_video_indices]

    if "participant_id" in base_df.columns:
        participant_ids = [
            base_df["participant_id"].tolist()[i] for i in existing_video_indices
        ]
    else:
        participant_ids = [0] * len(video_paths)

    sequence_ids = list(range(0, len(video_paths)))

    video_data = list(
        zip(
            video_paths,
            [results_path] * len(video_paths),
        )
    )
    with ProcessPoolExecutor(max_workers=workers) as executor:
        output_paths = list(
            tqdm(executor.map(process_video, *zip(*video_data)), total=len(video_data))
        )

    output_paths = glob.glob(f"{results_path}/landmark_files/*.parquet", recursive=True)
    df = pd.DataFrame(
        {
            "path": [output.replace("\\", "/").split("/")[-1] for output in output_paths],
            "participant_id": participant_ids,
            "sequence_id": sequence_ids,
            "sign": labels,
        }
    )
    df.to_csv(os.path.join(results_path, "landmarks.csv"), index=False)


if __name__ == "__main__":
    main()
