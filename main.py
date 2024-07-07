# %% imports############################
import uuid
from typing import Optional

import pandas as pd
from google.auth.transport.requests import Request
from google.cloud import aiplatform, storage
from google.cloud.aiplatform import CustomTrainingJob
from google.cloud.aiplatform.models import Model
from google.oauth2.service_account import Credentials

# %% args############################
key_filepath = None
project_id = "pioneering-flow-199508"
location = "europe-west4"
display_name = "pioneering-flow-199508-mnist-keras"
trainer_script_filepath = None
container_uri = "europe-docker.pkg.dev/vertex-ai/training/tf-gpu.2-13.py310:latest"  # "us-docker.pkg.dev/vertex-ai/training/tf-cpu.2-14.py310:latest"
bucket_name = "pioneering-flow-199508-c8713b72-1476-4bad-b6f1-bf6a47ca9926"
if not bucket_name:
    UUID = uuid.uuid4()
    print(UUID)
    bucket_name = f"{project_id}-{UUID}"


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


# %% Create a storage bucket############################
def create_bucket() -> storage.Bucket:
    storage_client = storage.Client(credentials=credentials, project=project_id)
    # get bucket if it does exist
    bucket = storage_client.lookup_bucket(bucket_name)
    if not bucket:
        bucket = storage_client.bucket(bucket_name)
        bucket.create(location=location)
    return bucket


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


# %% Create train job############################
def run_train_job() -> Optional[Model]:
    bucket = create_bucket()
    train_job = CustomTrainingJob(
        display_name=display_name,
        script_path=get_trainer_script_filepath(trainer_script_filepath),
        container_uri=container_uri,
        staging_bucket=f"gs://{bucket.name}",
        location=location,
    )
    model = train_job.run()
    return model


# %% Run bigquery query############################
def run_bq_query(sql: str) -> pd.DataFrame:
    bq_client = bigquery.Client(project=project_id, credentials=credentials)
    job_config = bigquery.QueryJobConfig()
    client_result = bq_client.query(sql, job_config=job_config)
    job_id = client_result.job_id
    df = client_result.result().to_arrow().to_pandas()
    print(f"Finished job_id: {job_id}")
    return df


# %% Run train job############################
credentials = get_credential(get_key_filepath(key_filepath))
print(credentials)

aiplatform.init(project=project_id, credentials=credentials)
model = run_train_job()
print(model)

# %% Delete bucket############################
bucket.delete(force=True)
