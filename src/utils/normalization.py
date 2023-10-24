import numpy as np
import pandas as pd
import click

template_length_per_column = {}


@click.command()
@click.option(
    "--pkl_path", default="annotations/final_dataset.pkl", help="Path to the .pkl file."
)
@click.option("--csv_path", default="missing_labels.csv", help="Path to the .csv file.")
@click.option(
    "--updated_pkl_path",
    default="annotations/updated_final_dataset.pkl",
    help="Path to save the updated .pkl file.",
)
def main(pkl_path, csv_path, updated_pkl_path):
    global template_length_per_column

    df_pkl = pd.read_pickle(pkl_path)

    for col in df_pkl.columns[3:]:
        non_empty_lists = (
            df_pkl[col]
            .apply(lambda x: len(x) if isinstance(x, list) and x else None)
            .dropna()
        )
        if not non_empty_lists.empty:
            template_length_per_column[col] = int(non_empty_lists.mean())

    df_pkl = df_pkl.apply(fill_empty_list_per_col, axis=1)

    for col in df_pkl.columns[3:]:
        df_pkl[col].apply(lambda cell: check_cell(cell, col))

    update_labels_in_pkl(df_pkl, csv_path, updated_pkl_path)

    print("Process completed.")


def fill_empty_list_per_col(row):
    for col in row.index:
        if isinstance(row[col], list):
            if not row[col]:
                if col in template_length_per_column:
                    row[col] = [0] * template_length_per_column[col]
                    print(f"Filled empty list in column {col}")
    return row


def check_cell(cell, col_name):
    if isinstance(cell, list):
        if not cell:
            print(f"Empty list found in column {col_name}")
        elif np.isnan(cell).any():
            print(f"NaN found in column {col_name}")


def update_labels_in_pkl(df_pkl, csv_path, updated_pkl_path):
    df_csv = pd.read_csv(csv_path)

    if "video_name" not in df_csv.columns or "label" not in df_csv.columns:
        print("The CSV must contain the 'video_name' and 'label' columns.")
        return

    for index, row in df_csv.iterrows():
        video_name = row["video_name"]
        new_label = row["label"]

        df_pkl.loc[df_pkl["video_name"] == video_name, "label"] = new_label

    zero_label_rows = df_pkl[df_pkl["label"] == 0]
    print("Lines with label=0:")
    print(zero_label_rows)

    df_pkl = df_pkl[df_pkl["label"] != 0]

    df_pkl.to_pickle(updated_pkl_path)

    print(f"Labels updated and saved in {updated_pkl_path}")


if __name__ == "__main__":
    main()
