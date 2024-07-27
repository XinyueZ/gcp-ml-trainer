import argparse
import os
import sys

import pandas as pd
from icecream import ic
from rich.pretty import pprint as pp
from sklearn.model_selection import train_test_split

this_file_dir = os.path.dirname(os.path.abspath(__file__))
this_file_dir_parent_dir = os.path.dirname(this_file_dir)
sys.path.append(this_file_dir)
sys.path.append(this_file_dir_parent_dir)
from base import TextBisonDatasetProcessor
from wine_process_base_dataset_processor import WinePriceBaseDatasetProcessor

INPUT_TEXT_PROMPT = """How much is the price of certain wine? Here is the information about the wine: The wine produced in the province of {0}, {1}, is described as '{2}'"""
OUTPUT_TEXT_PROMPT = """{0} US$"""


class WinePriceTextBisonDatasetProcessor(
    TextBisonDatasetProcessor, WinePriceBaseDatasetProcessor
):
    def __init__(
        self,
        csv_url: str,
        num_data_to_use: int,
        input_text_prompt: str,
        output_text_prompt: str,
        output_dir: str,
        train_set_filename: str,
        val_set_filename: str,
        test_size=0.2,
        random_state=42,
    ):
        super().__init__(output_dir, train_set_filename, val_set_filename)

        self.input_text_prompt = input_text_prompt
        self.output_text_prompt = output_text_prompt

        self.csv_url = csv_url
        self.num_data_to_use = num_data_to_use

        self.test_size = test_size
        self.random_state = random_state

    def apply_input_text_prompt(self, df: pd.DataFrame) -> pd.Series | pd.DataFrame:
        return df.apply(
            lambda row: self.input_text_prompt.format(
                row["province"], row["country"], row["description"]
            ),
            axis=1,
        )

    def apply_output_text_prompt(self, df: pd.DataFrame) -> pd.Series | pd.DataFrame:
        return df.apply(
            lambda row: self.output_text_prompt.format(row["price"]), axis=1
        )


"""
python wine_price_text_bison_dataset_processor.py \
    --csv_url "https://raw.githubusercontent.com/XinyueZ/llm-fine-tune-wine-price/master/data/wine_data.csv?token=GHSAT0AAAAAACACNBHDKU2RTW5IGQJKCYJSZLPTWMQ" \
    --num_data_to_use 1000 \
    --input_text_prompt "You are a super assistant that will be asked to help with the wine price prediction: How much is the price of certain wine? Here is the information about the wine: The wine produced in the province of {0}, {1}, is described as '{2}'" \
    --output_text_prompt "{0} US$" \
    --output_dir "/teamspace/studios/this_studio/gcp-ml-trainer/tmp" \
    --train_set_filename "ft_train_wine_price-21:24:07:2024.jsonl" \
    --val_set_filename "ft_val_wine_price-21:24:07:2024.jsonl" \
    --test_size 0.2 \
    --random_state 42

to subprocess call:
subprocess.run([
    'python', 'wine_price_text_bison_dataset_processor.py',
    '--csv_url', 'https://raw.githubusercontent.com/XinyueZ/llm-fine-tune-wine-price/master/data/wine_data.csv?token=GHSAT0AAAAAACACNBHDKU2RTW5IGQJKCYJSZLPTWMQ',
    '--num_data_to_use', '1000',
    '--input_text_prompt', 'You are a super assistant that will be asked to help with the wine price prediction: How much is the price of certain wine? Here is the information about the wine: The wine produced in the province of {0}, {1}, is described as \'{2}\'',
    '--output_text_prompt', '{0} US$',
    '--output_dir', '/teamspace/studios/this_studio/gcp-ml-trainer/tmp',
    '--train_set_filename', 'ft_train_wine_price-21:24:07:2024.jsonl',
    '--val_set_filename', 'ft_val_wine_price-21:24:07:2024.jsonl',
    '--test_size', '0.2',
    '--random_state', '42',
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
        "--input_text_prompt",
        type=str,
        default=INPUT_TEXT_PROMPT,
    )
    parser.add_argument(
        "--output_text_prompt",
        type=str,
        default=OUTPUT_TEXT_PROMPT,
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

    data_proc = WinePriceTextBisonDatasetProcessor(
        csv_url=args.csv_url,
        num_data_to_use=args.num_data_to_use,
        input_text_prompt=args.input_text_prompt,
        output_text_prompt=args.output_text_prompt,
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
