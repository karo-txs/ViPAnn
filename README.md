# ViPAnn - Video Landmark Processing Tool
### Description
This Python-based tool processes video files to extract and annotate body and hand landmarks. The output is saved as pkl files, each containing landmark coordinates and labels for individual videos. Additionally, the tool offers normalization utilities to adjust and clean your datasets.

### Requirements
```bash
    Python 3.x
    OpenCV
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
#### Video Processing
To process videos, run the following command:

```bash
python src/main.py \
    --dataset_name=<Dataset_Name> \
    --folder_path=<Folder_Path> \
    --output_folder=<Output_Folder> \
    --landmarks=<Landmarks> \
    --label_csv=<Label_CSV_Path> \
    --workers=<Workers> \
    --max_num_samples=<Max_Num_Samples> \
    --save_step=<Save_Step>
```

- Dataset_Name: The dataset name.
- Folder_Path: Directory containing videos to be processed.
- Output_Folder: Directory where annotated CSVs will be saved.
- Landmarks: List of landmarks to process (hand, body).
- Label_CSV_Path: Path to CSV file containing labels for videos (optional).
- Workers: Number of workers.
- Max_Num_Samples: Maximum number of videos to process.
- Save_Step: Step to partial save.

#### Data Normalization
To normalize your dataset (e.g., fill empty lists, update labels), run the following command:

```bash
python src/utils/normalization.py \
    --pkl_path=<PKL_Path> \
    --csv_path=<CSV_Path> \
    --updated_pkl_path=<Updated_PKL_Path>
```

- PKL_Path: Path to the .pkl file that you wish to normalize.
- CSV_Path: Path to the .csv file containing new labels.
- Updated_PKL_Path: Path where the updated .pkl file will be saved.

### Output

The output PKL files will be saved in the specified output folder, with filenames in the format:

```
<dataset_name>_<current_datetime>.pkl
```

Each CSV file contains the following columns:

- video_name: The name of the video file.
- frame_count: The total number of frames in the video.
- label: The label for the video (if labels are provided).
- Dynamic_Columns: These columns store the x and y coordinates for each landmark throughout the video.

#### Author

Karolayne Teixeira [karo-txs](https://github.com/karo-txs)
