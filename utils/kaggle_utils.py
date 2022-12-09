import os
import json
import subprocess
import zipfile
import glob


def register_creds(kaggle_creds_path: str):
    """Takes kaggle file from `kaggle_creds_path` and copies it in .kaggle directory."""
    KAGGLE_CONFIG_DIR = os.path.join(os.path.expandvars('$HOME'), '.kaggle')
    with open(kaggle_creds_path, "r", encoding='utf-8') as f:
        api_dict = json.load(fp=f)

    new_creds_path = f"{KAGGLE_CONFIG_DIR}/kaggle.json"
    with open(new_creds_path, "w", encoding='utf-8') as f:
        json.dump(api_dict, f)

    cmd = f"chmod 600 {KAGGLE_CONFIG_DIR}/kaggle.json"
    subprocess.check_output(cmd.split(" "))


def download_competition_files(competition_name: str, path_to_kaggle_creds: str, output_folder: str, unzip: bool = True):
    """
    !kaggle competitions download -c instacart-market-basket-analysis
    !unzip instacart-market-basket-analysis.zip -d /content/instacart-market-basket-analysis/
    """
    register_creds(path_to_kaggle_creds)

    import kaggle
    kaggle.api.competition_download_files(competition_name, path=output_folder)
    if unzip:
        zip_name = output_folder + "/" + competition_name + ".zip"
        new_output_folder = output_folder + "/" + competition_name + "/"
        with zipfile.ZipFile(zip_name, 'r') as zip_ref:
            zip_ref.extractall(new_output_folder)
        os.remove(zip_name)

        for data_zip in glob.glob(new_output_folder+"/*.csv.zip"):
            data_name = data_zip.rstrip(".zip")
            with zipfile.ZipFile(data_zip, 'r') as zip_ref:
                zip_ref.extractall(new_output_folder + "/" + data_name)
            os.remove(data_zip)
