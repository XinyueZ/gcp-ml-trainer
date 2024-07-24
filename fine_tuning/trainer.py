# %%
import argparse
import os
import sys
import warnings
from typing import Literal

from google.cloud import aiplatform
from google.cloud.aiplatform import PipelineJob
from icecream import ic
from kfp import compiler, dsl
from rich.pretty import pprint as pp

warnings.filterwarnings("ignore", category=FutureWarning, module="kfp.*")

this_file_dir = os.path.dirname(os.path.abspath(__file__))
this_file_dir_parent_dir = os.path.dirname(this_file_dir)
sys.path.append(this_file_dir)
sys.path.append(this_file_dir_parent_dir)

from base import Base
from utils import get_credential, get_datetime_now, get_key_filepath


class FineTuner(Base):
    job: PipelineJob
    pipeline_arguments: dict

    def __init__(
        self,
        model_mode: Literal["text", "chat"],
        model_display_name: str,
        location: str,
        project_id: str,
        training_steps: int,
        llm: str,
        accelerator_type: str,
        kfp_template_path: str,
        dataset_uri: str,
        evaluation_interval: int,
        evaluation_data_uri: str,
        key_dir: str,
        pipeline_root: str,
        enable_caching: bool = True,
    ):
        self.pipeline_arguments = {
            "model_display_name": model_display_name,
            "location": location,
            "large_model_reference": llm,
            "project": project_id,
            "train_steps": training_steps,
            "dataset_uri": dataset_uri,
            "accelerator_type": accelerator_type,
        }

        if model_mode == "text":
            pipeline_arguments["evaluation_interval"] = evaluation_interval
            pipeline_arguments["evaluation_data_uri"] = evaluation_data_uri

        self.job = PipelineJob(
            template_path=kfp_template_path,
            parameter_values=self.pipeline_arguments,
            pipeline_root=pipeline_root,
            enable_caching=enable_caching,
            display_name=model_display_name,
            location=location,
        )

    def apply(self):
        self.job.submit()

    def release(self):
        del self.job
        del self.pipeline_arguments


""" 
python trainer.py   --model_display_name "gcp_ft_wine_price_gemini-1.0-pro-001" \
                    --location "europe-west4" \
                    --project_id "isochrone-isodistance" \
                    --llm "gemini-1.0-pro-001" \
                    --accelerator_type "TPU" \
                    --kfp_template_path "https://us-kfp.pkg.dev/ml-pipeline/large-language-model-pipelines/tune-large-chat-model/v3.0.0" \
                    --dataset_uri "gs://isochrone-isodistance-ff943708-train//teamspace/studios/this_studio/gcp-ml-trainer/tmp/gemini_chat_ft_train_wine_price-21:24:07:2024.jsonl" \
                    --training_steps 1 \
                    --evaluation_interval 1 \
                    --evaluation_data_uri "gs://isochrone-isodistance-f609bba6-val//teamspace/studios/this_studio/gcp-ml-trainer/tmp/gemini_chat_ft_val_wine_price-21:24:07:2024.jsonl" \
                    --pipeline_root "gs://awesome-ml-ai/" \
                    --enable_caching 1 \
                    --model_mode "chat" 
"""
if __name__ == "__main__":
    # text tune pipeline：https://us-kfp.pkg.dev/ml-pipeline/large-language-model-pipelines/tune-large-model/v2.0.0
    # chat tune pipeline：https://us-kfp.pkg.dev/ml-pipeline/large-language-model-pipelines/tune-large-chat-model/v3.0.0
    # doc: https://cloud.google.com/vertex-ai/generative-ai/docs/models/tune-code-models?hl=zh-cn
    # llm = "gemini-1.0-pro-002"  # ? not yet avaiable at gcp: "gemini-1.5-flash"

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--key_dir",
        type=str,
        required=False,
        default=os.path.join(this_file_dir_parent_dir, "keys"),
    )
    parser.add_argument("--model_display_name", type=str, required=True)
    parser.add_argument("--location", type=str, required=False, default="europe-west4")
    parser.add_argument("--project_id", type=str, required=True)
    parser.add_argument("--training_steps", type=int, required=False, default=1)
    parser.add_argument("--llm", type=str, required=False, default="gemini-1.0-pro-002")
    parser.add_argument("--accelerator_type", type=str, required=False, default="TPU")
    parser.add_argument(
        "--kfp_template_path",
        type=str,
        required=False,
        default="https://us-kfp.pkg.dev/ml-pipeline/large-language-model-pipelines/tune-large-chat-model/v3.0.0",
    )
    parser.add_argument(
        "--dataset_uri",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--evaluation_interval",
        type=int,
        required=False,
        default=1,
    )
    parser.add_argument(
        "--evaluation_data_uri",
        type=str,
        required=False,
    )
    parser.add_argument(
        "--pipeline_root",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--enable_caching",
        type=int,
        required=False,
        default=1,
    )
    parser.add_argument(
        "--model_mode",
        type=str,
        required=True,
        choices=["text", "chat"],
        default="chat",
    )

    args = parser.parse_args()

    credentials = get_credential(get_key_filepath(key_dir=args.key_dir))
    ic(f"Credentials: {credentials}")

    aiplatform.init(project=args.project_id, credentials=credentials)
    ic("Initialized AI Platform")

    fine_tuner = FineTuner(
        model_mode=args.model_mode,
        model_display_name="{}_{}".format(args.model_display_name, get_datetime_now()),
        location=args.location,
        project_id=args.project_id,
        training_steps=args.training_steps,
        llm=args.llm,
        accelerator_type=args.accelerator_type,
        kfp_template_path=args.kfp_template_path,
        dataset_uri=args.dataset_uri,
        evaluation_interval=args.evaluation_interval,
        evaluation_data_uri=args.evaluation_data_uri,
        key_dir=args.key_dir,
        pipeline_root=args.pipeline_root,
        enable_caching=args.enable_caching == 1,
    )

    fine_tuner.apply()
    pp(fine_tuner.job.state)
    fine_tuner.release()
