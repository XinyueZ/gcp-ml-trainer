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
    df_train: pd.DataFrame
    df_val: pd.DataFrame
    train_set_output_filefullpath: str
    val_set_output_filefullpath: str
    model_name: str
    mode: str

    def __init__(
        self,
        model_name: str,
        mode: str,
        output_dir: str,
        train_set_filename: str,
        val_set_filename: str,
    ):
        self.model_name = model_name
        self.mode = mode

        train_set_output_filename = "{0}_{1}_{2}".format(
            model_name, mode, train_set_filename
        )
        val_set_output_filename = "{0}_{1}_{2}".format(
            model_name, mode, val_set_filename
        )

        self.train_set_output_filefullpath = os.path.join(
            output_dir, train_set_output_filename
        )
        self.val_set_output_filefullpath = os.path.join(
            output_dir, val_set_output_filename
        )

    def create_df(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        raise NotImplementedError

    def create_jsonl(self, df: pd.DataFrame, filefullpath: str) -> str:
        raise NotImplementedError

    def apply(self):
        self.df_train, self.df_val = self.create_df()
        self.create_jsonl(self.df_train, self.train_set_output_filefullpath)
        self.create_jsonl(self.df_val, self.val_set_output_filefullpath)

    def release(self):
        del self.df_train
        del self.df_val


class TextBisonDatasetProcessor(BaseProcessor):
    def __init__(
        self,
        output_dir: str,
        train_set_filename: str,
        val_set_filename: str,
    ):
        super().__init__(
            model_name="text-bison",
            mode="text",
            output_dir=output_dir,
            train_set_filename=train_set_filename,
            val_set_filename=val_set_filename,
        )

    def create_jsonl(self, df: pd.DataFrame, filefullpath: str):
        """
        # create json-lines, every line is a json object like below:
        https://cloud.google.com/vertex-ai/generative-ai/docs/models/tune-text-models-supervised?hl=en#chat
        ....
        {"input_text": "hello", "output_text": "world"}
        ....
        """
        cols = ["input_text", "output_text"]
        tune_jsonl = df[cols].to_json(orient="records", lines=True)
        with open(filefullpath, "w") as f:
            f.write(tune_jsonl)


class ChatBisonDatasetProcessor(BaseProcessor):
    def __init__(
        self,
        context_prompt: str,
        output_dir: str,
        train_set_filename: str,
        val_set_filename: str,
    ):
        super().__init__(
            model_name="chat-bison",
            mode="chat",
            output_dir=output_dir,
            train_set_filename=train_set_filename,
            val_set_filename=val_set_filename,
        )

        self.context_prompt = context_prompt

    def create_jsonl(self, df: pd.DataFrame, filefullpath: str):
        """
        # create json-lines, every line is a json object like below:
        # Gemini fine-tune dataset rule: https://cloud.google.com/vertex-ai/generative-ai/docs/models/tune-text-models-supervised?hl=en#chat
        #
                {
                "context": "You are a pirate dog named Captain Barktholomew.",
                "messages": [
                    {
                    "author": "user",
                    "content": "Hi"
                    },
                    {
                    "author": "assistant",
                    "content": "Argh! What brings ye to my ship?"
                    }
                ]
                }
        """
        with open(filefullpath, "w") as f:
            df.apply(
                lambda row: f.write(
                    f'{{"context": "{self.context_prompt}", "messages": [{{"author": "user", "content": "{row["input_text"]}"}}'
                    f',{{"author": "assistant", "content": "{row["output_text"]}"}}]}}\n'
                ),
                axis=1,
            )


class GeminiChatDatasetProcessor(BaseProcessor):
    def __init__(
        self,
        sys_prompt: str,
        output_dir: str,
        train_set_filename: str,
        val_set_filename: str,
    ):
        super().__init__(
            model_name="gemini",
            mode="chat",
            output_dir=output_dir,
            train_set_filename=train_set_filename,
            val_set_filename=val_set_filename,
        )

        self.sys_prompt = sys_prompt

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
        """
        with open(filefullpath, "w") as f:
            df.apply(
                lambda row: f.write(
                    f'{{"messages": [{{"role": "system", "content": "{self.sys_prompt}"}},{{"role": "user", "content": "{row["input_text"]}"}}'
                    f',{{"role": "model", "content": "{row["output_text"]}"}}]}}\n'
                ),
                axis=1,
            )
