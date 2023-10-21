# ViPAnn - Video Landmark Processing Tool
### Description
This Python-based tool processes video files to extract and annotate body and hand landmarks. The output is saved as CSV files, each containing landmark coordinates and labels for individual videos.

### Requirements
```bash
    Python 3.x
    OpenCV
    Matplotlib
    Pandas
    Tqdm
    Click
    MediaPipe (util functions)
```

### Installation
Clone this repository:

```bash
git clone https://github.com/karo-txs/ViPAnn.git
```

Navigate to the project directory and install the required packages:

```bash
cd ViPAnn
pip install -r requirements.txt
```

### Usage
To process videos, run the following command:

```bash
python main.py --dataset_name=<Dataset_Name> --folder_path=<Folder_Path> --output_folder=<Output_Folder> --landmarks=<Landmarks> --label_csv=<Label_CSV_Path> --save_sample_rate=<Save_Sample_Rate>
```
- Dataset_Name: The dataset name.
- Folder_Path: Directory containing videos to be processed.
- Output_Folder: Directory where annotated CSVs will be saved.
- Landmarks: List of landmarks to process (hand, body).
- Label_CSV_Path: Path to CSV file containing labels for videos (optional).
- Save_Sample_Rate: Rate at which sample frames will be saved as .png. Default is 100.

### Output

The output CSVs will be saved in the specified output folder, with filenames in the format:

```
<dataset_name>_<current_datetime>.csv
```

Each CSV file contains the following columns:

- video_name: The name of the video file.
- frame_count: The total number of frames in the video.
- label: The label for the video (if labels are provided).
- Dynamic_Columns: These columns store the x and y coordinates for each landmark throughout the video.

#### Author

Karolayne Silva [karo-txs](https://github.com/karo-txs)