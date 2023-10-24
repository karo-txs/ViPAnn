import unittest
import pandas as pd
import numpy as np

class TestDatasetIntegrity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = pd.read_pickle('annotations/updated_final_dataset.pkl')
        print(cls.df.head())
        
    def test_unique_labels(self):
        unique_labels = self.df['label'].nunique() if 'label' in self.df.columns else 'Column not found'
        self.assertNotEqual(unique_labels, 'Column not found', "Label column not found in dataset")
    
    def test_label_counts(self):
        if 'label' in self.df.columns:
            label_counts = self.df['label'].value_counts()
        else:
            label_counts = 'Column not found'
            
        self.assertNotEqual(str(label_counts), 'Column not found', "Label column not found in dataset")
    
    def test_mean_of_columns(self):
        def calculate_list_mean(cell):
            if isinstance(cell, list):
                if not cell:
                    return 0
                elif np.isnan(cell).any():
                    return 0
                return np.mean(cell)
            return cell

        def check_cell(cell, col_name):
            if isinstance(cell, list):
                if not cell:
                    print(f"Empty list encountered at column {col_name}")
                elif np.isnan(cell).any():
                    print(f"NaN encountered at column {col_name}")

        selected_cols = self.df.columns[3:]

        for col in selected_cols:
            self.df[col].apply(lambda cell: check_cell(cell, col))

        df_means = self.df[selected_cols].map(calculate_list_mean)
        overall_means = df_means.mean()
        print(overall_means)

    def test_list_videos_with_zero_label(self):
        zero_label_rows = self.df[self.df['label'] == 0]
        zero_label_count = len(zero_label_rows)
        
        if zero_label_count > 0:
            video_names = zero_label_rows['video_name'].tolist()
            self.fail(f"Found {zero_label_count} videos with label=0. Videos: {video_names}")

if __name__ == '__main__':
    unittest.main()
