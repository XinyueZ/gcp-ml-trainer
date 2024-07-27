import argparse
import os
import sys

import pandas as pd
from icecream import ic
from rich.pretty import pprint as pp
from sklearn.model_selection import train_test_split

this_file_dir = os.path.dirname(os.path.abspath(__file__))
this_file_root_dir = os.path.dirname(os.path.dirname(this_file_dir))
ic(this_file_root_dir)
sys.path.append(this_file_dir)
sys.path.append(this_file_root_dir)
from base import BaseDatasetProcessor


class WinePriceBaseDatasetProcessor(BaseDatasetProcessor):

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
        train_df = self.create_insturct_columns(train_df)
        eval_df = self.create_insturct_columns(eval_df)
        ic(len(train_df), len(eval_df))
        return train_df, eval_df
