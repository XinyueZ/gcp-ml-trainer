# %%
import os
import sys
import warnings

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

from utils import (get_credential, get_datetime_now, get_key_filepath)

# %%
model_display_name = "gcp_ft_wine_price_gemini-1.0-pro-002"
location = "europe-west4"
project_id = "isochrone-isodistance"
training_steps = 3
llm = "gemini-1.0-pro-002"  # ? not yet avaiable at gcp: "gemini-1.5-flash"
accelerator_type = "TPU"
# text gen：https://us-kfp.pkg.dev/ml-pipeline/large-language-model-pipelines/tune-large-model/v2.0.0
# chat gen：https://us-kfp.pkg.dev/ml-pipeline/large-language-model-pipelines/tune-large-chat-model/v3.0.0
# doc: https://cloud.google.com/vertex-ai/generative-ai/docs/models/tune-code-models?hl=zh-cn
kfp_template_path = "https://us-kfp.pkg.dev/ml-pipeline/large-language-model-pipelines/tune-large-chat-model/v3.0.0"
dataset_uri = "gs://isochrone-isodistance-14b31089-train//teamspace/studios/this_studio/gcp-ml-trainer/tmp/ft_train_wine_price-21:23:07:2024.jsonl"
# validation is optional for text-gen not for chat-gen
evaluation_data_uri = "gs://pioneering-flow-199508-11ef3e5e-eval//teamspace/studios/this_studio/gcp-ml-trainer/tmp/ft_eval_wine_price-20:23:07:2024.jsonl"
evaluation_interval = 3
key_dir = os.path.join(this_file_dir_parent_dir, "keys")
pipeline_root = "gs://awesome-ml-ai/"

# %%
credentials = get_credential(get_key_filepath(key_dir=key_dir))
ic(f"Credentials: {credentials}")

aiplatform.init(project=project_id, credentials=credentials)
ic("Initialized AI Platform")

ic("")

# %%
pipeline_arguments = {
    "model_display_name": model_display_name,
    "location": location,
    "large_model_reference": llm,
    "project": project_id,
    "train_steps": training_steps,
    "dataset_uri": dataset_uri,
    "accelerator_type": accelerator_type,
    # The two validations below are optional for text-gen not for chat-gen
    # "evaluation_interval": evaluation_interval,
    # "evaluation_data_uri": evaluation_data_uri,
}

# %%
job = PipelineJob(
    template_path=kfp_template_path,
    parameter_values=pipeline_arguments,
    pipeline_root=pipeline_root,
    enable_caching=True,
    display_name=model_display_name,
    location=location,
)

# %% submit for execution
job.submit()

# %% check to see the status of the job
pp(job.state)

# %%
