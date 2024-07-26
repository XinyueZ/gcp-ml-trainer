# %%
import argparse
import os
import sys
from subprocess import PIPE, CompletedProcess, Popen, run

import wine_price_chat_bison_dataset_processor
import wine_price_gemini_chat_dataset_processor
import wine_price_text_bison_dataset_processor
from icecream import ic
from rich.pretty import pprint as pp

this_file_dir = os.path.dirname(os.path.abspath(__file__))
this_file_dir_parent_dir = os.path.dirname(this_file_dir)
sys.path.append(this_file_dir)
sys.path.append(this_file_dir_parent_dir)

from utils import get_datetime_now


def create_dataset(
    model_name: str,
    data_url: str,
    num_data_to_use: int,
    dataset_output_dir: str,
    train_set_filename: str,
    val_set_filename: str,
) -> CompletedProcess:
    match model_name:
        case "gemini-chat":
            ic(
                model_name,
                data_url,
                num_data_to_use,
                dataset_output_dir,
                train_set_filename,
                val_set_filename,
            )
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
                "0.2",
                "--random_state",
                "42",
            ]

        case "chat-bison":
            ic(
                model_name,
                data_url,
                num_data_to_use,
                dataset_output_dir,
                train_set_filename,
                val_set_filename,
            )
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
                "0.2",
                "--random_state",
                "42",
            ]

        case "text-bison":
            ic(
                model_name,
                data_url,
                num_data_to_use,
                dataset_output_dir,
                train_set_filename,
                val_set_filename,
            )
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
                "0.2",
                "--random_state",
                "42",
            ]

        case _:
            raise ValueError("model_name not found")
    return run(cmd, capture_output=True, text=True)


"""
python wine_price_data_pipe.py --model_name gemini-chat
python wine_price_data_pipe.py --model_name chat-bison
python wine_price_data_pipe.py --model_name text-bison
"""
if __name__ == "__main__":
    # never changed
    tsmp = get_datetime_now()
    train_set_filename = "ft_train_wine_price-{}.jsonl".format(tsmp)
    val_set_filename = "ft_val_wine_price-{}.jsonl".format(tsmp)
    ic(train_set_filename), ic(val_set_filename)

    # args
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        type=str,
        default="gemini-chat",
        choices=[
            "gemini-chat",
            "chat-bison",
            "text-bison",
        ],
    )
    parser.add_argument(
        "--num_data_to_use",
        type=int,
        required=False,
        default=1000,
    )
    args = parser.parse_args()

    # create data for train and val sets, dfferent project is here a little bit different
    data_url = "https://raw.githubusercontent.com/XinyueZ/llm-fine-tune-wine-price/master/data/wine_data.csv?token=GHSAT0AAAAAACACNBHDKU2RTW5IGQJKCYJSZLPTWMQ"
    dataset_output_dir = os.path.join(this_file_dir_parent_dir, "tmp")
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
    pusher_dir = os.path.join(this_file_dir_parent_dir, "bucket")
    pusher_cmd = os.path.join(pusher_dir, "bucket_pusher.py")
    os.system(
        "python {}  --project_id isochrone-isodistance "
        "--file_fullpath {} --bucket_name_postfix {}_train".format(
            pusher_cmd,
            train_set_filefullpath,
            args.model_name,
        )
    )
    os.system(
        "python {}  --project_id isochrone-isodistance "
        "--file_fullpath {} --bucket_name_postfix {}_val".format(
            pusher_cmd,
            val_set_filefullpath,
            args.model_name,
        )
    )
