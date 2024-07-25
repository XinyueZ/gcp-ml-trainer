# %%
import argparse
import os
import sys

from icecream import ic
from rich.pretty import pprint as pp
from sklearn.model_selection import train_test_split
from wine_price_gemini_chat_dataset_processor import (
    SYS_PROMPT, WinePriceGeminiChatDatasetProcessor)
from wine_price_text_bison_dataset_processor import \
    WinePriceTextBisonDatasetProcessor

this_file_dir = os.path.dirname(os.path.abspath(__file__))
this_file_dir_parent_dir = os.path.dirname(this_file_dir)
sys.path.append(this_file_dir)
sys.path.append(this_file_dir_parent_dir)

from utils import get_datetime_now

"""
python wine_price_data_pipe.py --dataset_type gemini
python wine_price_data_pipe.py --dataset_type text-bison
"""
if __name__ == "__main__":
    data_url = "https://raw.githubusercontent.com/XinyueZ/llm-fine-tune-wine-price/master/data/wine_data.csv?token=GHSAT0AAAAAACACNBHDKU2RTW5IGQJKCYJSZLPTWMQ"
    dataset_output_dir = os.path.join(this_file_dir_parent_dir, "tmp")

    tsmp = get_datetime_now()
    num_data_to_use = 1000

    train_set_filename = "ft_train_wine_price-{}.jsonl".format(tsmp)
    val_set_filename = "ft_val_wine_price-{}.jsonl".format(tsmp)
    ic(train_set_filename), ic(val_set_filename)

    # args, ie. dataset_type
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_type",
        type=str,
        default="gemini",
        choices=["gemini", "text-bison"],
    )
    args = parser.parse_args()
    dataset_type = args.dataset_type
    ic(dataset_type)

    # process data for train and val sets
    match dataset_type:
        case "gemini":
            from wine_price_gemini_chat_dataset_processor import (AI_PROMPT,
                                                                  HUMAN_PROMPT)
        case "text-bison":
            from wine_price_text_bison_dataset_processor import (AI_PROMPT,
                                                                 HUMAN_PROMPT)

    dataset_type_dict = {
        "gemini": WinePriceGeminiChatDatasetProcessor(
            csv_url=data_url,
            num_data_to_use=num_data_to_use,
            sys_prompt=SYS_PROMPT,
            human_prompt=HUMAN_PROMPT,
            ai_prompt=AI_PROMPT,
            output_dir=dataset_output_dir,
            train_set_filename=train_set_filename,
            val_set_filename=val_set_filename,
        ),
        "text-bison": WinePriceTextBisonDatasetProcessor(
            csv_url=data_url,
            num_data_to_use=num_data_to_use,
            human_prompt=HUMAN_PROMPT,
            ai_prompt=AI_PROMPT,
            output_dir=dataset_output_dir,
            train_set_filename=train_set_filename,
            val_set_filename=val_set_filename,
        ),
    }

    data_proc = dataset_type_dict[dataset_type]
    data_proc.apply()
    data_proc.release()

    train_set_filefullpath = data_proc.train_set_output_filefullpath
    val_set_filefullpath = data_proc.val_set_output_filefullpath
    ic(train_set_filefullpath, val_set_filefullpath)

    # run pusher like:
    """
    python dataset_bucket_pusher.py  --project_id "isochrone-isodistance" \
                     --file_fullpath /teamspace/studios/this_studio/gcp-ml-trainer/tmp/gemini_chat_ft_train_wine_price-21:24:07:2024.jsonl \
                     --bucket_name_postfix "train"

    python dataset_bucket_pusher.py  --project_id "isochrone-isodistance" \
                     --file_fullpath /teamspace/studios/this_studio/gcp-ml-trainer/tmp/gemini_chat_ft_val_wine_price-21:24:07:2024.jsonl \
                     --bucket_name_postfix "val"
    """
    os.system(
        "python dataset_bucket_pusher.py  --project_id isochrone-isodistance "
        "--file_fullpath {} --bucket_name_postfix train".format(train_set_filefullpath)
    )
    os.system(
        "python dataset_bucket_pusher.py  --project_id isochrone-isodistance "
        "--file_fullpath {} --bucket_name_postfix val".format(val_set_filefullpath)
    )
