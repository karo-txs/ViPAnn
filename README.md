# ViPAnn - Video Landmark Processing Tool
### Description
This project provides a tool to extract landmarks from videos using the MediaPipe Holistic model and structures the data to be consistent with the Google Kaggle competition on ASL Signs found here.

### Requirements
```
Python 3.8 or above
pandas
OpenCV
MediaPipe
Click
```
    
You can install these requirements using pip:

```bash
pip install -r requirements.txt
```

### Usage:
#### Generate the dataset:
To extract landmarks from your videos and generate the train data, run the following command:

```bash
python src/vipann.py --base_file C:/Users/karol/Downloads/ufv/ufv.csv --video_path C:/Users/karol/Downloads/ufv/ --results_path ufv --workers 1
```

Where:
- path_to_base.csv is the path to your base CSV file containing video names and labels.
- path_to_videos_folder is the directory path containing all the videos.
- path_to_save_results (optional) is the directory where you want to save the results. Defaults to ./results.
- number_of_workers (optional) is the number of parallel workers you want to use for processing. Defaults to 4.

##### Input Structure:
- Videos: Videos should be stored in a directory. Each video corresponds to a sequence of ASL signs.
- Support CSV:
    - sign: Name of the sign shown in the video.
    - path: Filename of the video.

##### Structure of generated files:
- Landmark Data (train_landmark_files/[participant_id]/[sequence_id].parquet):
    - frame: The frame number in the raw video.
    - row_id: A unique identifier for the row.
    - type: The type of landmark. One of ['face', 'left_hand', 'pose', 'right_hand'].
    - landmark_index: The landmark index number.
    - x/y/z: The normalized spatial coordinates of the landmark.

- Landmarks Data (landmarks.csv):
    - path: The path to the landmark file.
    - participant_id: A unique identifier for the data contributor.
    - sequence_id: A unique identifier for the landmark sequence.
    - sign: The label for the landmark sequence.

#### Test the generated dataset:
a. Using pytest:
```bash
pytest src/test.py
```
b. Using the command line:
```bash
python src/test.py --result-path results_test --base-file-path vlibrasil_bilingual_v2.csv
```

#### Notes:
- The data structure is designed to match the [Google Kaggle competition on ASL Signs](https://www.kaggle.com/competitions/asl-signs/data).
- Ensure the videos are properly lit to maximize the accuracy of landmark detection.
- The MediaPipe model is not fully trained to predict depth, so the Z-values in the landmark data may not be completely accurate.

#### Author

Karolayne Teixeira [karo-txs](https://github.com/karo-txs)
