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


class BaseDatasetProcessor(Base):
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

    def create_insturct_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        df["input_text"] = self.apply_input_text_prompt(df)
        df["output_text"] = self.apply_output_text_prompt(df)

        return df

    def apply_input_text_prompt(self, df: pd.DataFrame) -> pd.Series | pd.DataFrame:
        raise NotImplementedError

    def apply_output_text_prompt(self, df: pd.DataFrame) -> pd.Series | pd.DataFrame:
        raise NotImplementedError

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


class GemmaInstructDatasetProcessor(BaseDatasetProcessor):
    def __init__(
        self,
        output_dir: str,
        train_set_filename: str,
        val_set_filename: str,
    ):
        super().__init__(
            model_name="gemma-instruct",
            mode="text",
            output_dir=output_dir,
            train_set_filename=train_set_filename,
            val_set_filename=val_set_filename,
        )

    def create_jsonl(self, df: pd.DataFrame, filefullpath: str):
        """
        # create instruction tuned model with format, every line is a json object like below:
        https://www.kaggle.com/models/keras/gemma2
        https://ai.google.dev/gemma/docs/formatting

        start_of_turn_user = "<start_of_turn>user\n"
        start_of_turn_model = "<start_of_turn>model\n"
        end_of_turn = "<end_of_turn>\n"
        prompt = start_of_turn_user + input_text + end_of_turn + start_of_turn_model + output_text + end_of_turn

        ....
        <start_of_turn>user
        hello<end_of_turn>
        <start_of_turn>model
        world<end_of_turn>
        ....
        """
        cols = ["input_text", "output_text"]
        start_of_turn_user = "<start_of_turn>user\n"
        start_of_turn_model = "<start_of_turn>model\n"
        end_of_turn = "<end_of_turn>\n"
        with open(filefullpath, "w") as f:
            df.apply(
                lambda row: f.write(
                    f'{start_of_turn_user}{row["input_text"]}{end_of_turn}{start_of_turn_model}{row["output_text"]}{end_of_turn}'
                ),
                axis=1,
            )


class TextBisonDatasetProcessor(BaseDatasetProcessor):
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


class ChatBisonDatasetProcessor(BaseDatasetProcessor):
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


class GeminiChatDatasetProcessor(BaseDatasetProcessor):
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


class AgentBuilderDatasetProcessor(BaseDatasetProcessor):
    def __init__(
        self,
        output_dir: str,
        train_set_filename: str,
        val_set_filename: str,
    ):
        super().__init__(
            model_name="agent-builder-unstruct",
            mode="text",
            output_dir=output_dir,
            train_set_filename=train_set_filename,
            val_set_filename=val_set_filename,
        )

    def create_jsonl(self, df: pd.DataFrame, filefullpath: str):
        """
        # create unstructed text dataset as txt file.
        # Google Agent Builder unstruct dataset (for agent app): https://cloud.google.com/generative-ai-app-builder/docs/prepare-data#unstructured
        #
        # content of "input_text"
        # content of "output_text"
        """
        cols = ["input_text", "output_text"]
        with open(filefullpath, "w") as f:
            df.apply(
                lambda row: f.write(f'{row["input_text"]}\n{row["output_text"]}\n\n'),
                axis=1,
            )


class DialogFlowDatasetProcessor(BaseDatasetProcessor):
    def __init__(
        self,
        output_dir: str,
        train_set_filename: str,
        val_set_filename: str,
    ):

        super().__init__(
            model_name="dialogflow-struct",
            mode="chat",
            output_dir=output_dir,
            train_set_filename=train_set_filename,
            val_set_filename=val_set_filename,
        )

    def create_jsonl(self, df: pd.DataFrame, filefullpath: str):
        """
        # create structed CSV dataset.
        # Google Agent Builder struct dataset (for dialogflow chat app): https://cloud.google.com/dialogflow/vertex/docs/concept/data-store#structured
        #
        # two columns CSV file, 1st column is "input_text", 2nd column is "output_text".
        """
        cols = ["input_text", "output_text"]
        with open(filefullpath, "w") as f:
            # header: question, answer
            f.write("question,answer\n")
            df.apply(
                lambda row: f.write(f'"{row["input_text"]}",{row["output_text"]}\n'),
                axis=1,
            )
