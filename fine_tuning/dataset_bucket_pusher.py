import argparse
import os
import sys
import uuid

from google.cloud import aiplatform, storage
from google.oauth2.service_account import Credentials
from icecream import ic
from rich.pretty import pprint as pp

this_file_dir = os.path.dirname(os.path.abspath(__file__))
this_file_dir_parent_dir = os.path.dirname(this_file_dir)
sys.path.append(this_file_dir)
sys.path.append(this_file_dir_parent_dir)

from base import Base
from utils import get_credential, get_key_filepath


class DatasetBucketPusher(Base):

    def __init__(
        self,
        project_id: str,
        location: str,
        bucket_name: str,
        credentials: Credentials,
        file_fullpath: str,  # local location of the object to push
    ):
        self.credentials = credentials
        self.project_id = project_id
        self.location = location
        self.bucket_name = bucket_name
        self.file_fullpath = file_fullpath

    def _create_bucket(self) -> storage.Bucket:
        storage_client = storage.Client(
            credentials=self.credentials, project=self.project_id
        )
        # get bucket if it does exist
        bucket = storage_client.lookup_bucket(self.bucket_name)
        if not bucket:
            bucket = storage_client.bucket(self.bucket_name)
            bucket.create(location=self.location, predefined_acl="publicRead")
        return bucket

    def _push2bucket(self):
        self.bucket = self._create_bucket()
        self.blob = self.bucket.blob(self.file_fullpath)
        self.blob.upload_from_filename(self.file_fullpath)

    def apply(self) -> storage.Blob:
        self._push2bucket()
        return self.blob

    def release(self):
        del self.blob
        del self.bucket


"""
python dataset_bucket_pusher.py  --project_id "isochrone-isodistance" \
                     --file_fullpath /teamspace/studios/this_studio/gcp-ml-trainer/tmp/gemini_chat_ft_train_wine_price-21:24:07:2024.jsonl \
                     --bucket_name_postfix "train"

python dataset_bucket_pusher.py  --project_id "isochrone-isodistance" \
                     --file_fullpath /teamspace/studios/this_studio/gcp-ml-trainer/tmp/gemini_chat_ft_val_wine_price-21:24:07:2024.jsonl \
                     --bucket_name_postfix "val"

python dataset_bucket_pusher.py  --project_id "isochrone-isodistance" \
                     --file_fullpath /teamspace/studios/this_studio/gcp-ml-trainer/tmp/text-bison@001_text_ft_train_wine_price-10:25:07:2024.jsonl \
                     --bucket_name_postfix "train"

python dataset_bucket_pusher.py  --project_id "isochrone-isodistance" \
                     --file_fullpath /teamspace/studios/this_studio/gcp-ml-trainer/tmp/text-bison@001_text_ft_val_wine_price-10:25:07:2024.jsonl \
                     --bucket_name_postfix "val"
"""
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--key_dir",
        type=str,
        required=False,
        default=os.path.join(this_file_dir_parent_dir, "keys"),
    )
    parser.add_argument("--project_id", type=str, required=True)
    parser.add_argument("--location", type=str, required=False, default="europe-west4")
    parser.add_argument("--file_fullpath", type=str, required=True)
    parser.add_argument("--bucket_name_postfix", type=str, required=True)
    args = parser.parse_args()

    buuid = uuid.uuid4()
    buuid = str(buuid)[:8]
    ic(buuid)
    bucket_name = "{0}-{1}-{2}".format(
        args.project_id,
        buuid,
        args.bucket_name_postfix,
    )
    ic(bucket_name)

    credentials = get_credential(get_key_filepath(key_dir=args.key_dir))
    ic(f"Credentials: {credentials}")

    dp = DatasetBucketPusher(
        args.project_id,
        args.location,
        bucket_name,
        credentials,
        args.file_fullpath,
    )
    blob = dp.apply()
    ic(blob)
    ic(blob.public_url)
    dp.release()
