import os
import sys

import pandas as pd


class Base:
    def __call__(self):
        raise NotImplementedError("Not implemented, call apply()")

    def apply(self):
        raise NotImplementedError

    def release(self):
        raise NotImplementedError


class BaseProcessor(Base):

    def create_df(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        raise NotImplementedError

    def create_jsonl(self, df: pd.DataFrame, filefullpath: str) -> str:
        raise NotImplementedError

    def apply(self, inputs):
        raise NotImplementedError


class GeminiChatDatasetProcessor(BaseProcessor):
    def __init__(
        self, sys_prompt: str, output_dir: str, train_filename: str, val_filename: str
    ):
        model_name = "gemini"
        mode = "chat"
        self.sys_prompt = sys_prompt

        train_output_filename = "{0}_{1}_{2}".format(model_name, mode, train_filename)
        val_output_filename = "{0}_{1}_{2}".format(model_name, mode, val_filename)

        self.train_output_filefullpath = os.path.join(output_dir, train_output_filename)
        self.val_output_filefullpath = os.path.join(output_dir, val_output_filename)

    def create_jsonl(self, df: pd.DataFrame, filefullpath: str):
        """
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
        # every row, the  "input_text" is for the "content" of "user".
        #            the  "output_text" is for the "content" of "model".
        # the sys_prompt is for the "content" of "system".
        """
        with open(filefullpath, "w") as f:
            df.apply(
                lambda row: f.write(
                    f'{{"messages": [{{"role": "system", "content": "{self.sys_prompt}"}},{{"role": "user", "content": "{row["input_text"]}"}}'
                    f',{{"role": "model", "content": "{row["output_text"]}"}}]}}\n'
                ),
                axis=1,
            )

    def apply(self):
        self.df_train, self.df_val = self.create_df()
        self.create_jsonl(self.df_train, self.train_output_filefullpath)
        self.create_jsonl(self.df_val, self.val_output_filefullpath)

    def release(self):
        del self.df_train
        del self.df_val
