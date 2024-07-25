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
from base import TextBisonTextDatasetProcessor

HUMAN_PROMPT = """How much is the price of certain wine? Here is the information about the wine: The wine produced in the province of {0}, {1}, is described as '{2}'"""
AI_PROMPT = """{0} US$"""


class WinePriceTextBisonDatasetProcessor(TextBisonTextDatasetProcessor):
    def __init__(
        self,
        csv_url: str,
        num_data_to_use: int,
        human_prompt: str,
        ai_prompt: str,
        output_dir: str,
        train_set_filename: str,
        val_set_filename: str,
        test_size=0.2,
        random_state=42,
    ):
        super().__init__(output_dir, train_set_filename, val_set_filename)
        model_name = "gemini"
        mode = "chat"

        self.human_prompt = human_prompt
        self.ai_prompt = ai_prompt

        self.csv_url = csv_url
        self.num_data_to_use = num_data_to_use

        self.test_size = test_size
        self.random_state = random_state

    def _create_insturct_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        df["input_text"] = df.apply(
            lambda row: self.human_prompt.format(
                row["province"], row["country"], row["description"]
            ),
            axis=1,
        )
        df["output_text"] = df.apply(
            lambda row: self.ai_prompt.format(row["price"]), axis=1
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
