import pandas as pd
import click
import os


@click.command()
@click.option("--result-path", help="Path to the results file.", required=True)
@click.option("--base-file-path", help="Path to the base.csv file.", required=True)
def run_tests(result_path, base_file_path):
    result_csv_path = f"{result_path}/landmarks.csv"
    test_data_integrity(result_path, result_csv_path, base_file_path)
    test_value_range(result_path, result_csv_path)
    test_signs_integrity(result_csv_path, base_file_path)
    print("All tests passed!")


def find_file(target_file, search_path):
    for dirpath, dirnames, filenames in os.walk(search_path):
        if target_file in filenames:
            return os.path.join(dirpath, target_file)
    return None


def test_data_integrity(result_path, result_csv_path, base_file_path):
    train_df = pd.read_csv(result_csv_path)
    base_df = pd.read_csv(base_file_path)

    assert not train_df.isnull().values.any(), "Missing values in result.csv"

    for path in train_df["path"]:
        assert find_file(path, result_path), f"Missing .parquet file for path: {path}"


def test_value_range(result_path, result_csv_path):
    train_df = pd.read_csv(result_csv_path)

    for path in train_df["path"]:
        df = pd.read_parquet(find_file(path, result_path))
        assert df["x"].between(0.0, 1.0).all(), f"x values out of range in {path}"
        assert df["y"].between(0.0, 1.0).all(), f"y values out of range in {path}"


def test_signs_integrity(result_csv_path, base_file_path):
    train_df = pd.read_csv(result_csv_path)
    base_df = pd.read_csv(base_file_path)

    unique_signs_in_train = train_df["sign"].unique()
    unique_signs_in_base = base_df["label"].unique()

    print(len(unique_signs_in_train))
    print(len(unique_signs_in_base))

    assert len(unique_signs_in_train) == len(
        unique_signs_in_base
    ), "Signs in train.csv not found in base.csv"


if __name__ == "__main__":
    run_tests()
