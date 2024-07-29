import argparse
import os
import sys
from subprocess import PIPE, CompletedProcess, Popen, run

import wine_price_chat_bison_dataset_processor
import wine_price_gemini_chat_dataset_processor
import wine_price_gemma_instruct_dataset_processor
import wine_price_text_bison_dataset_processor
from icecream import ic
from rich.pretty import pprint as pp
from utils import get_datetime_now


def create_dataset(
    model_name: str,
    data_url: str,
    num_data_to_use: int,
    dataset_output_dir: str,
    train_set_filename: str,
    val_set_filename: str,
    test_size=0.2,
    random_state=42,
) -> CompletedProcess:

    ic(
        model_name,
        data_url,
        num_data_to_use,
        dataset_output_dir,
        train_set_filename,
        val_set_filename,
    )
    match model_name:
        case "gemini-chat":
            cmd = [
                "python",
                "wine_price_gemini_chat_dataset_processor.py",
                "--csv_url",
                data_url,
                "--num_data_to_use",
                str(num_data_to_use),
                "--sys_prompt",
                wine_price_gemini_chat_dataset_processor.SYS_PROMPT,
                "--user_prompt",
                wine_price_gemini_chat_dataset_processor.USER_PROMPT,
                "--model_prompt",
                wine_price_gemini_chat_dataset_processor.MODEL_PROMPT,
                "--output_dir",
                dataset_output_dir,
                "--train_set_filename",
                train_set_filename,
                "--val_set_filename",
                val_set_filename,
                "--test_size",
                str(test_size),
                "--random_state",
                str(random_state),
            ]

        case "gemma-instruct":
            cmd = [
                "python",
                "wine_price_gemma_instruct_dataset_processor.py",
                "--csv_url",
                data_url,
                "--num_data_to_use",
                str(num_data_to_use),
                "--user_prompt",
                wine_price_gemma_instruct_dataset_processor.USER_PROMPT,
                "--model_prompt",
                wine_price_gemma_instruct_dataset_processor.MODEL_PROMPT,
                "--output_dir",
                dataset_output_dir,
                "--train_set_filename",
                train_set_filename,
                "--val_set_filename",
                val_set_filename,
                "--test_size",
                str(test_size),
                "--random_state",
                str(random_state),
            ]
        case "chat-bison":
            cmd = [
                "python",
                "wine_price_chat_bison_dataset_processor.py",
                "--csv_url",
                data_url,
                "--num_data_to_use",
                str(num_data_to_use),
                "--context_prompt",
                wine_price_chat_bison_dataset_processor.CONTEXT_PROMPT,
                "--user_prompt",
                wine_price_chat_bison_dataset_processor.USER_PROMPT,
                "--assistant_prompt",
                wine_price_chat_bison_dataset_processor.ASSISTANT_PROMPT,
                "--output_dir",
                dataset_output_dir,
                "--train_set_filename",
                train_set_filename,
                "--val_set_filename",
                val_set_filename,
                "--test_size",
                str(test_size),
                "--random_state",
                str(random_state),
            ]

        case "text-bison":
            cmd = [
                "python",
                "wine_price_text_bison_dataset_processor.py",
                "--csv_url",
                data_url,
                "--num_data_to_use",
                str(num_data_to_use),
                "--input_text_prompt",
                wine_price_text_bison_dataset_processor.INPUT_TEXT_PROMPT,
                "--output_text_prompt",
                wine_price_text_bison_dataset_processor.OUTPUT_TEXT_PROMPT,
                "--output_dir",
                dataset_output_dir,
                "--train_set_filename",
                train_set_filename,
                "--val_set_filename",
                val_set_filename,
                "--test_size",
                str(test_size),
                "--random_state",
                str(random_state),
            ]

        case _:
            raise ValueError("model_name not found")
    return run(cmd, capture_output=True, text=True)


"""
python pipeline.py --key_dir "OAuth2" --project_id "isochrone-isodistance" --predefined_acl "projectPrivate" --location "europe-west1" --model_name "gemini-chat"
python pipeline.py --key_dir "OAuth2" --project_id "isochrone-isodistance" --predefined_acl "projectPrivate" --location "europe-west1" --model_name "gemma-instruct"
python pipeline.py --key_dir "OAuth2" --project_id "isochrone-isodistance" --predefined_acl "projectPrivate" --location "europe-west1" --model_name "chat-bison"  
python pipeline.py --key_dir "OAuth2" --project_id "isochrone-isodistance" --predefined_acl "projectPrivate" --location "europe-west1" --model_name "text-bison"  
"""
if __name__ == "__main__":
    this_file_dir = os.path.dirname(os.path.abspath(__file__))
    this_file_root_dir = os.path.dirname(os.path.dirname(this_file_dir))
    # never changed
    tsmp = get_datetime_now()
    train_set_filename = "ft_train_wine_price-{}.jsonl".format(tsmp)
    val_set_filename = "ft_val_wine_price-{}.jsonl".format(tsmp)
    ic(train_set_filename), ic(val_set_filename)

    # args
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--key_dir",
        type=str,
        required=False,
        help="Set 'OAuth2' or it is a path of the dir of service account key json.",
        default=os.path.join(this_file_root_dir, "keys"),
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="gemini-chat",
        choices=[
            "gemini-chat",
            "gemma-instruct",
            "chat-bison",
            "text-bison",
        ],
    )
    parser.add_argument(
        "--project_id",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--predefined_acl",
        type=str,
        required=False,
        default="projectPrivate",
        help="https://cloud.google.com/storage/docs/access-control/lists#predefined-acl",
    )
    parser.add_argument(
        "--num_data_to_use",
        type=int,
        required=False,
        default=1000,
    )
    parser.add_argument(
        "--location",
        type=str,
        required=False,
        default="europe-west1",
    )
    parser.add_argument("--bucket_name", type=str, required=False)
    args = parser.parse_args()

    # create data for train and val sets, dfferent project is here a little bit different
    data_url = "https://raw.githubusercontent.com/XinyueZ/llm-fine-tune-wine-price/master/data/wine_data.csv?token=GHSAT0AAAAAACACNBHDKU2RTW5IGQJKCYJSZLPTWMQ"
    dataset_output_dir = os.path.join(this_file_root_dir, "tmp")
    result = create_dataset(
        model_name=args.model_name,
        data_url=data_url,
        num_data_to_use=args.num_data_to_use,
        dataset_output_dir=dataset_output_dir,
        train_set_filename=train_set_filename,
        val_set_filename=val_set_filename,
    )
    results = result.stdout.strip().split(";")
    ic(results)
    train_set_filefullpath = results[0]
    val_set_filefullpath = results[1]
    ic(train_set_filefullpath, val_set_filefullpath)

    # push to gcp bucket, should never be changed
    pusher_dir = os.path.join(this_file_root_dir, "bucket")
    pusher_cmd = os.path.join(pusher_dir, "bucket_pusher.py")
    train_cmd = "python {} --key_dir {} --project_id {} --predefined_acl {} --file_fullpath {} --location {} --cate train ".format(
        pusher_cmd,
        args.key_dir,
        args.project_id,
        args.predefined_acl,
        train_set_filefullpath,
        args.location,
    )
    if args.bucket_name:
        train_cmd += "--bucket_name {}".format(args.bucket_name)
    os.system(train_cmd)

    val_cmd = "python {}  --key_dir {} --project_id {} --predefined_acl {} --file_fullpath {} --location {} --cate val ".format(
        pusher_cmd,
        args.key_dir,
        args.project_id,
        args.predefined_acl,
        val_set_filefullpath,
        args.location,
    )
    if args.bucket_name:
        val_cmd += "--bucket_name {}".format(args.bucket_name)
    os.system(val_cmd)
