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


class BucketPusher(Base):

    def __init__(
        self,
        project_id: str,
        location: str,
        bucket_name: str,
        credentials: Credentials,
        file_fullpath: str,  # local location of the object to push
        predefined_acl: str = "projectPrivate",  # https://cloud.google.com/storage/docs/access-control/lists#predefined-acl
    ):
        self.credentials = credentials
        self.project_id = project_id
        self.location = location
        self.bucket_name = bucket_name
        self.file_fullpath = file_fullpath
        self.predefined_acl = predefined_acl

    def _create_bucket(self) -> storage.Bucket:
        storage_client = storage.Client(
            credentials=self.credentials, project=self.project_id
        )
        # get bucket if it does exist
        bucket = storage_client.lookup_bucket(self.bucket_name)
        if not bucket:
            bucket = storage_client.bucket(self.bucket_name)
            bucket.create(location=self.location, predefined_acl=self.predefined_acl)
        return bucket

    def _push2bucket(self):
        self.bucket = self._create_bucket()
        self.blob = self.bucket.blob(os.path.basename(self.file_fullpath))
        self.blob.upload_from_filename(self.file_fullpath)

    def apply(self) -> storage.Blob:
        self._push2bucket()
        return self.blob

    def release(self):
        del self.blob
        del self.bucket


"""
python bucket_pusher.py  --project_id "isochrone-isodistance" \
                     --key_dir "OAuth2" \
                     --predefined_acl "projectPrivate" \
                     --location "europe-west1" \
                     --file_fullpath /teamspace/studios/this_studio/gcp-ml-trainer/tmp/gemini_chat_ft_train_wine_price-12:25:07:2024.jsonl \
                     --bucket_name_postfix "train" 

python bucket_pusher.py  --project_id "isochrone-isodistance" \
                     --key_dir "OAuth2" \
                     --predefined_acl  "projectPrivate" \
                     --location "europe-west1" \
                     --file_fullpath /teamspace/studios/this_studio/gcp-ml-trainer/tmp/gemini_chat_ft_val_wine_price-12:25:07:2024.jsonl \
                     --bucket_name_postfix "val"

python bucket_pusher.py  --project_id "isochrone-isodistance" \
                     --key_dir "OAuth2" \
                     --predefined_acl  "projectPrivate" \
                     --location "europe-west1" \
                     --file_fullpath /teamspace/studios/this_studio/gcp-ml-trainer/tmp/text-bison@001_text_ft_train_wine_price-11:25:07:2024.jsonl \
                     --bucket_name_postfix "train"

python bucket_pusher.py  --project_id "isochrone-isodistance" \
                     --key_dir "OAuth2" \
                     --predefined_acl  "projectPrivate" \
                     --location "europe-west1" \
                     --file_fullpath /teamspace/studios/this_studio/gcp-ml-trainer/tmp/text-bison@001_text_ft_val_wine_price-11:25:07:2024.jsonl \
                     --bucket_name_postfix "val"
"""
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--key_dir",
        type=str,
        required=False,
        help="Set 'OAuth2' or it is a path of the dir of service account key json.",
        default=os.path.join(this_file_dir_parent_dir, "keys"),
    )
    parser.add_argument("--project_id", type=str, required=True)
    parser.add_argument("--location", type=str, required=False, default="europe-west1")
    parser.add_argument("--file_fullpath", type=str, required=True)
    parser.add_argument("--bucket_name_postfix", type=str, required=True)
    parser.add_argument(
        "--predefined_acl",
        type=str,
        required=False,
        default="projectPrivate",
        help="https://cloud.google.com/storage/docs/access-control/lists#predefined-acl",
    )

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

    credentials = None
    if args.key_dir != "OAuth2":
        credentials = get_credential(get_key_filepath(key_dir=args.key_dir))
        ic(f"Credentials: {credentials}")

    dp = BucketPusher(
        project_id=args.project_id,
        location=args.location,
        bucket_name=bucket_name,
        credentials=credentials,
        file_fullpath=args.file_fullpath,
        predefined_acl=args.predefined_acl,
    )
    blob = dp.apply()
    ic(blob)
    ic(blob.public_url)
    dp.release()
