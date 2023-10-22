import pandas as pd

def update_labels_in_pkl(pkl_path, csv_path, updated_pkl_path):
    df_pkl = pd.read_pickle(pkl_path)
    
    df_csv = pd.read_csv(csv_path)
    
    if 'video_name' not in df_csv.columns or 'label' not in df_csv.columns:
        print("CSV must contain 'video_name' and 'label' columns.")
        return
    
    for index, row in df_csv.iterrows():
        video_name = row['video_name']
        new_label = row['label']
        
        df_pkl.loc[df_pkl['video_name'] == video_name, 'label'] = new_label
    
    zero_label_rows = df_pkl[df_pkl['label'] == 0]
    print("Lines with label=0:")
    print(zero_label_rows)
    
    df_pkl = df_pkl[df_pkl['label'] != 0]
    
    df_pkl.to_pickle(updated_pkl_path)

    print(f"Labels updated and saved to {updated_pkl_path}")

pkl_path = 'annotations/final_dataset.pkl'
csv_path = 'missing_labels.csv'  # CSV with columns 'video_name' e 'label'
updated_pkl_path = 'annotations/updated_final_dataset.pkl'

update_labels_in_pkl(pkl_path, csv_path, updated_pkl_path)
