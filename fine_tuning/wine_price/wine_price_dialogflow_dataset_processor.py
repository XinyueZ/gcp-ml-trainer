import argparse
import os
import sys

import pandas as pd
from icecream import ic
from rich.pretty import pprint as pp
from sklearn.model_selection import train_test_split
from wine_price_base_dataset_processor import WinePriceBaseDatasetProcessor
from base import DialogFlowDatasetProcessor

QUESTION_PROMPT = """The wine produced in the province of {0}, {1}, is described as '{2}'"""
ANSWER_PROMPT = """{0} US$"""


class WinePriceDialogFlowDatasetProcessor(
    DialogFlowDatasetProcessor, WinePriceBaseDatasetProcessor
):
    def __init__(
        self,
        csv_url: str,
        num_data_to_use: int,
        question_prompt: str,
        answer_prompt: str,
        output_dir: str,
        train_set_filename: str,
        val_set_filename: str,
        test_size=0.2,
        random_state=42,
    ):
        super().__init__(output_dir, train_set_filename, val_set_filename)

        self.question_prompt = question_prompt
        self.answer_prompt = answer_prompt

        self.csv_url = csv_url
        self.num_data_to_use = num_data_to_use

        self.test_size = test_size
        self.random_state = random_state

    def apply_input_text_prompt(self, df: pd.DataFrame) -> pd.Series | pd.DataFrame:
        return df.apply(
            lambda row: self.question_prompt.format(
                row["province"], row["country"], row["description"]
            ),
            axis=1,
        )

    def apply_output_text_prompt(self, df: pd.DataFrame) -> pd.Series | pd.DataFrame:
        return df.apply(
            lambda row: self.answer_prompt.format(row["price"]), axis=1
        )


"""
python wine_price_dialogflow_dataset_processor.py  \
    --csv_url "https://raw.githubusercontent.com/XinyueZ/llm-fine-tune-wine-price/master/data/wine_data.csv?token=GHSAT0AAAAAACACNBHDKU2RTW5IGQJKCYJSZLPTWMQ" \
    --num_data_to_use 1000 \
    --question_prompt "The wine produced in the province of {0}, {1}, is described as '{2}'" \
    --answer_prompt "{0} US$" \
    --output_dir "/teamspace/studios/this_studio/gcp-ml-trainer/tmp" \
    --train_set_filename "ft_train_wine_price.csv" \
    --val_set_filename "ft_val_wine_price.csv" \
    --test_size 0.2 \
    --random_state 42

to subprocess call:
subprocess.run([
    'python', 'wine_price_dialogflow_dataset_processor.py',
    '--csv_url', 'https://raw.githubusercontent.com/XinyueZ/llm-fine-tune-wine-price/master/data/wine_data.csv?token=GHSAT0AAAAAACACNBHDKU2RTW5IGQJKCYJSZLPTWMQ',
    '--num_data_to_use', '1000' ,
    '--question_prompt', "The wine produced in the province of {0}, {1}, is described as '{2}'" ,
    '--answer_prompt', "{0} US$" ,
    '--output_dir', '/teamspace/studios/this_studio/gcp-ml-trainer/tmp' ,
    '--train_set_filename', 'ft_train_wine_price.csv' ,
    '--val_set_filename', 'ft_val_wine_price.csv' ,
    '--test_size', '0.2' ,
    '--random_state', '42'
])
"""
if __name__ == "__main__":
    ic(sys.argv)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv_url",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--num_data_to_use",
        type=int,
        default=1000,
    )
    parser.add_argument(
        "--question_prompt",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--answer_prompt",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--train_set_filename",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--val_set_filename",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--test_size",
        type=float,
        default=0.2,
    )
    parser.add_argument(
        "--random_state",
        type=int,
        default=42,
    )
    args = parser.parse_args()
    ic(args)

    data_proc = WinePriceDialogFlowDatasetProcessor(
        csv_url=args.csv_url,
        num_data_to_use=args.num_data_to_use,
        question_prompt=args.question_prompt,
        answer_prompt=args.answer_prompt,
        output_dir=args.output_dir,
        train_set_filename=args.train_set_filename,
        val_set_filename=args.val_set_filename,
        test_size=args.test_size,
        random_state=args.random_state,
    )
    data_proc.apply()
    data_proc.release()

    print(
        f"{data_proc.train_set_output_filefullpath};{data_proc.val_set_output_filefullpath}"
    )
