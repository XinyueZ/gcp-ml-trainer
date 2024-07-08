import os

from google.oauth2.service_account import Credentials


# %% Do credential refresh############################
def get_credential(key_path: str) -> Credentials:
    credentials = Credentials.from_service_account_file(
        key_path, scopes=["https://www.googleapis.com/auth/cloud-platform"]
    )

    if credentials.expired:
        credentials.refresh(Request())
    return credentials


# %% Get credential file############################
def get_key_filepath(key_filepath: str = None) -> str:
    if key_filepath is None:
        this_file_dir = os.path.dirname(os.path.abspath(__file__))
        # set root to this file dir
        os.chdir(this_file_dir)
        find_json_file = lambda path: [
            f for f in os.listdir(path) if f.endswith(".json")
        ][0]
        full_path = os.path.join("./keys/", find_json_file("./keys/"))
        return full_path
    else:
        return key_filepath


# %% Get train script############################
def get_trainer_script_filepath(script_filepath: str) -> str:
    if script_filepath is None:
        this_file_dir = os.path.dirname(os.path.abspath(__file__))
        # set root to this file dir
        os.chdir(this_file_dir)
        find_py_file = lambda path: [f for f in os.listdir(path) if f.endswith(".py")][
            0
        ]
        full_path = os.path.join("./train_script/", find_py_file("./train_script/"))
        return full_path
    else:
        return script_filepath
