# %% imports############################
import uuid
from typing import Optional

import pandas as pd
from base import Base
from google.auth.transport.requests import Request
from google.cloud import aiplatform, storage
from google.cloud.aiplatform import CustomTrainingJob
from google.cloud.aiplatform.models import Model
from google.oauth2.service_account import Credentials
from utils import get_credential, get_key_filepath, get_trainer_script_filepath

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


class Trainer(Base):

    credentials: Credentials
    project_id: str
    bucket: storage.Bucket

    def __init__(self, credentials: Credentials, project_id: str):
        self.credentials = credentials
        self.project_id = project_id

    def create_bucket(self) -> storage.Bucket:
        storage_client = storage.Client(
            credentials=self.credentials, project=self.project_id
        )
        # get bucket if it does exist
        bucket = storage_client.lookup_bucket(bucket_name)
        if not bucket:
            bucket = storage_client.bucket(bucket_name)
            bucket.create(location=location)
        self.bucket = bucket
        return bucket

    def apply(self) -> Optional[Model]:
        bucket = self.create_bucket()
        train_job = CustomTrainingJob(
            display_name=display_name,
            script_path=get_trainer_script_filepath(trainer_script_filepath),
            container_uri=container_uri,
            staging_bucket=f"gs://{bucket.name}",
            location=location,
        )
        model = train_job.run()
        return model

    def release(self):
        self.bucket.delete(force=True)


def main():
    credentials = get_credential(get_key_filepath(key_filepath))
    print(f"Credentials: {credentials}")

    aiplatform.init(project=project_id, credentials=credentials)
    print("Initialized AI Platform")

    trainer = Trainer(credentials, project_id)
    model = trainer.apply()
    print(model)

    trainer.release()
    print("Released resources")

 
if __name__ == "__main__":
    main()
