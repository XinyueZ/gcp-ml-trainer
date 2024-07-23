# %%
import os
import sys
import uuid

import pandas as pd
from google.cloud import aiplatform, storage
from icecream import ic
from rich.pretty import pprint as pp
from sklearn.model_selection import train_test_split

this_file_dir = os.path.dirname(os.path.abspath(__file__))
this_file_dir_parent_dir = os.path.dirname(this_file_dir)
sys.path.append(this_file_dir)
sys.path.append(this_file_dir_parent_dir)

from utils import (get_credential, get_datetime_now, get_key_filepath,
                   get_trainer_script_filepath)

# %%
location = "europe-west4"
key_dir = os.path.join(this_file_dir_parent_dir, "keys")
project_id = "isochrone-isodistance"
num_data_to_use = 1000
bucket_name = "{0}-{1}-{2}"

# %%
sys_prompt = (
    "You are a super assistant. You are asked to help with the wine price prediction."
)
human_prompt = """How much is the price of certain wine? Here is the information about the wine: The wine produced in the province of {0}, {1}, is described as '{2}'"""
ai_message = """{0} US$."""
s_test = human_prompt.format(
    "California",
    "US",
    "Aromas include tropical fruit, broom, brimstone and dried herb.",
)
ss_test = ai_message.format("235.0")
ic(s_test)
ic(ss_test)
ic("")


# %%
url = "https://raw.githubusercontent.com/XinyueZ/llm-fine-tune-wine-price/master/data/wine_data.csv?token=GHSAT0AAAAAACACNBHDKU2RTW5IGQJKCYJSZLPTWMQ"
df = pd.read_csv(url)
df.head()

# %%
df = df.dropna(how="any")
df = df.loc[(df != "NaN").any(axis=1)]
df = df.loc[:, ["province", "country", "description", "price"]]
df["price"] = df["price"].astype(str)
df = df.reset_index(drop=True)
df = df[df["country"] == "US"]
df = df.head(num_data_to_use)
df.head()

# %%
# check len of df
len(df)

# %%
# split data
train_df, eval_df = train_test_split(df, test_size=0.2, random_state=42)


# %%
def create_insturction_columns(df: pd.DataFrame) -> pd.DataFrame:
    df["input_text"] = df.apply(
        lambda row: human_prompt.format(
            row["province"], row["country"], row["description"]
        ),
        axis=1,
    )
    df["output_text"] = df.apply(lambda row: ai_message.format(row["price"]), axis=1)
    return df


train_df = create_insturction_columns(train_df)
eval_df = create_insturction_columns(eval_df)

# %%
train_df.head()

# %%
eval_df.head()


# %%
# create json-lines, every line is a json object like below:
# Gemini fine-tune dataset rule: https://cloud.google.com/vertex-ai/generative-ai/docs/models/gemini-supervised-tuning-about
#
# {
#     "messages": [
#               {
#                   "role": "system",
#                   "content": "You are a pirate dog named Captain Barktholomew."
#               },
#               {
#                   "role": "user",
#                   "content": "Hi"
#               },
#               {
#                   "role": "model",
#                   "content": "Argh! What brings ye to my ship?"
#               }
#     ]
# }
def create_jsonl(df: pd.DataFrame, filefullpath: str) -> str:
    # every row, the  "input_text" is for the "content" of "user".
    #            the  "output_text" is for the "content" of "model".
    # the sys_prompt is for the "content" of "system".
    with open(filefullpath, "w") as f:
        for index, row in df.iterrows():
            f.write(
                f'{{"messages": [{{"role": "system", "content": "{sys_prompt}"}},{{"role": "user", "content": "{row["input_text"]}"}}'
                f',{{"role": "model", "content": "{row["output_text"]}"}}]}}\n'
            )
    return filefullpath


tsmp = get_datetime_now()
train_filefullpath = create_jsonl(
    train_df,
    os.path.join(this_file_dir_parent_dir, "tmp", f"ft_train_wine_price-{tsmp}.jsonl"),
)
eval_filefullpath = create_jsonl(
    eval_df,
    os.path.join(this_file_dir_parent_dir, "tmp", f"ft_eval_wine_price-{tsmp}.jsonl"),
)
ic(train_filefullpath), ic(eval_filefullpath)

# %%
credentials = get_credential(get_key_filepath(key_dir=key_dir))
ic(f"Credentials: {credentials}")

aiplatform.init(project=project_id, credentials=credentials)
ic("Initialized AI Platform")

ic("")

# %%
uuid = uuid.uuid4()
# only first 8 characters
uuid = str(uuid)[:8]
ic(uuid)
ic("")


# %%
def create_bucket(bucket_name: str) -> storage.Bucket:
    storage_client = storage.Client(credentials=credentials, project=project_id)
    # get bucket if it does exist
    bucket = storage_client.lookup_bucket(bucket_name)
    if not bucket:
        bucket = storage_client.bucket(bucket_name)
        bucket.create(location=location)
    bucket = bucket
    return bucket


train_bucket_name = bucket_name.format(
    project_id,
    uuid,
    "train",
)
eval_bucket_name = bucket_name.format(
    project_id,
    uuid,
    "eval",
)
train_bucket = create_bucket(train_bucket_name)
eval_bucket = create_bucket(eval_bucket_name)


# %%
def push2bucket(bucket: storage.Bucket, file_fullpath: str) -> storage.Blob:
    blob = bucket.blob(file_fullpath)
    blob.upload_from_filename(file_fullpath)
    return blob


blob_train = push2bucket(train_bucket, train_filefullpath)
blob_eval = push2bucket(eval_bucket, eval_filefullpath)

# %%
ic(blob_train.public_url)
ic(blob_eval.public_url)
ic("")
