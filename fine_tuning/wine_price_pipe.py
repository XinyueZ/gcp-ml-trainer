# %%
import argparse
import os
import sys

from icecream import ic
from rich.pretty import pprint as pp
from sklearn.model_selection import train_test_split
from wine_price_dataset_processor import (AI_PROMPT, HUMAN_PROMPT, SYS_PROMPT,
                                          WinePriceGeminiChatDatasetProcessor)

this_file_dir = os.path.dirname(os.path.abspath(__file__))
this_file_dir_parent_dir = os.path.dirname(this_file_dir)
sys.path.append(this_file_dir)
sys.path.append(this_file_dir_parent_dir)

from utils import get_datetime_now

if __name__ == "__main__":
    data_url = "https://raw.githubusercontent.com/XinyueZ/llm-fine-tune-wine-price/master/data/wine_data.csv?token=GHSAT0AAAAAACACNBHDKU2RTW5IGQJKCYJSZLPTWMQ"
    dataset_output_dir = os.path.join(this_file_dir_parent_dir, "tmp")

    tsmp = get_datetime_now()
    num_data_to_use = 1000

    train_set_filename = "ft_train_wine_price-{}.jsonl".format(tsmp)
    val_set_filename = "ft_val_wine_price-{}.jsonl".format(tsmp)
    ic(train_set_filename), ic(val_set_filename)

    data_proc = WinePriceGeminiChatDatasetProcessor(
        csv_url=data_url,
        num_data_to_use=num_data_to_use,
        sys_prompt=SYS_PROMPT,
        human_prompt=HUMAN_PROMPT,
        ai_prompt=AI_PROMPT,
        output_dir=dataset_output_dir,
        train_set_filename=train_set_filename,
        val_set_filename=val_set_filename,
    )
    data_proc.apply()
    data_proc.release()
