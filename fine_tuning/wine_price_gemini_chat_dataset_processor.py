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
from base import GeminiChatDatasetProcessor

SYS_PROMPT = (
    "You are a super assistant. You are asked to help with the wine price prediction."
)
USER_PROMPT = """How much is the price of certain wine? Here is the information about the wine: The wine produced in the province of {0}, {1}, is described as '{2}'"""
MODEL_PROMPT = """{0} US$"""


class WinePriceGeminiChatDatasetProcessor(GeminiChatDatasetProcessor):
    def __init__(
        self,
        csv_url: str,
        num_data_to_use: int,
        sys_prompt: str,
        user_prompt: str,
        model_prompt: str,
        output_dir: str,
        train_set_filename: str,
        val_set_filename: str,
        test_size=0.2,
        random_state=42,
    ):
        super().__init__(sys_prompt, output_dir, train_set_filename, val_set_filename)

        self.user_prompt = user_prompt
        self.model_prompt = model_prompt

        self.csv_url = csv_url
        self.num_data_to_use = num_data_to_use

        self.test_size = test_size
        self.random_state = random_state

    def _create_insturct_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        df["input_text"] = df.apply(
            lambda row: self.user_prompt.format(
                row["province"], row["country"], row["description"]
            ),
            axis=1,
        )
        df["output_text"] = df.apply(
            lambda row: self.model_prompt.format(row["price"]), axis=1
        )
        return df

    def create_df(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        df = pd.read_csv(self.csv_url)
        df = df.dropna(how="any")
        df = df.loc[(df != "NaN").any(axis=1)]
        df = df.loc[:, ["province", "country", "description", "price"]]
        df["price"] = df["price"].astype(str)
        df = df.reset_index(drop=True)
        df = df[df["country"] == "US"]
        df = df.head(self.num_data_to_use)
        train_df, eval_df = train_test_split(
            df, test_size=self.test_size, random_state=self.random_state
        )
        train_df = self._create_insturct_columns(train_df)
        eval_df = self._create_insturct_columns(eval_df)
        ic(len(train_df), len(eval_df))
        return train_df, eval_df

"""
python wine_price_gemini_chat_dataset_processor.py \
    --csv_url "https://raw.githubusercontent.com/XinyueZ/llm-fine-tune-wine-price/master/data/wine_data.csv?token=GHSAT0AAAAAACACNBHDKU2RTW5IGQJKCYJSZLPTWMQ" \
    --num_data_to_use 1000 \
    --sys_prompt "You are a super assistant. You are asked to help with the wine price prediction." \
    --user_prompt "How much is the price of certain wine? Here is the information about the wine: The wine produced in the province of {0}, {1}, is described as '{2}'" \
    --model_prompt "{0} US$" \
    --output_dir "/teamspace/studios/this_studio/gcp-ml-trainer/tmp" \
    --train_set_filename "ft_train_wine_price-2021-10-12T15:00:00.jsonl" \
    --val_set_filename "ft_val_wine_price-2021-10-12T15:00:00.jsonl" \
    --test_size 0.2 \
    --random_state 42

to subprocess call:
subprocess.run(
    [
        "python",
        "wine_price_gemini_chat_dataset_processor.py",
        "--csv_url",
        "https://raw.githubusercontent.com/XinyueZ/llm-fine-tune-wine-price/master/data/wine_data.csv?token=GHSAT0AAAAAACACNBHDKU2RTW5IGQJKCYJSZLPTWMQ",
        "--num_data_to_use",
        "1000",
        "--sys_prompt",
        "You are a super assistant. You are asked to help with the wine price prediction.",
        "--user_prompt",
        "How much is the price of certain wine? Here is the information about the wine: The wine produced in the province of {0}, {1}, is described as '{2}'",
        "--model_prompt",
        "{0} US$",
        "--output_dir",
        "tmp",
        "--train_set_filename",
        "ft_train_wine_price-2021-10-12T15:00:00.jsonl",
        "--val_set_filename",
        "ft_val_wine_price-2021-10-12T15:00:00.jsonl",
        "--test_size",
        "0.2",
        "--random_state",
        "42",
    ]
)
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
        "--sys_prompt",
        type=str,
        default=SYS_PROMPT,
    )
    parser.add_argument(
        "--user_prompt",
        type=str,
        default=USER_PROMPT,
    )
    parser.add_argument(
        "--model_prompt",
        type=str,
        default=MODEL_PROMPT,
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

    data_proc = WinePriceGeminiChatDatasetProcessor(
        csv_url=args.csv_url,
        num_data_to_use=args.num_data_to_use,
        sys_prompt=args.sys_prompt,
        user_prompt=args.user_prompt,
        model_prompt=args.model_prompt,
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
