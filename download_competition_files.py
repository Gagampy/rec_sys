from fire import Fire
from utils.kaggle_utils import download_competition_files


def download_competition(
        name: str, output_folder: str = "./data/", path_to_kaggle_creds: str = "./aux_files/kaggle.json"
):
    download_competition_files(
        competition_name=name, path_to_kaggle_creds=path_to_kaggle_creds, output_folder=output_folder
    )


if __name__ == "__main__":
    Fire(download_competition)
