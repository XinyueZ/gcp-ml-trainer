import functools
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
from google.cloud import aiplatform
from google.oauth2.service_account import Credentials
from icecream import ic

ic.configureOutput(includeContext=True, contextAbsPath=True)
import functools
import math
import time
from concurrent.futures import ThreadPoolExecutor
from typing import List

import numpy as np
from base import Base
from loguru import logger
from sklearn.metrics.pairwise import cosine_similarity
from tqdm.auto import tqdm
from utils import get_credential, get_key_filepath, get_trainer_script_filepath
from vertexai.language_models import TextEmbeddingModel


class TextEmbedder(Base):
    model_name: str
    model: TextEmbeddingModel

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = TextEmbeddingModel.from_pretrained(model_name)

    def release(self):
        del self.model

    def embed(self, inputs: str | Sequence[str]) -> List["TextEmbedding"]:
        vectors = self.model.get_embeddings(inputs)
        return vectors

    def embed_texts(self, sentences: Sequence[str]) -> np.ndarray | None:
        try:
            vectors = self.embed(sentences)
            return np.asarray([vector.values for vector in vectors])
        except Exception:
            return None

    def _generate_batches(self, sentences: Sequence[str], batch_size=5) -> List[str]:
        for i in range(0, len(sentences), batch_size):
            yield sentences[i : i + batch_size]

    def embed_texts_in_batches(
        self,
        sentences: Sequence[str],
        api_calls_per_second=0.33,
        batch_size=5,
    ) -> np.ndarray:
        embeddings_list = []
        batch_generator = self._generate_batches(sentences, batch_size)
        call_frequence = 1 / api_calls_per_second
        with ThreadPoolExecutor() as executor:
            futures = []
            for batch in tqdm(
                batch_generator,
                total=math.ceil(len(sentences) / batch_size),
                position=0,
            ):
                futures.append(
                    executor.submit(functools.partial(self.embed_texts), batch)
                )
                time.sleep(call_frequence)

            for future in futures:
                embeddings_list.extend(future.result())

        is_successful = [
            embedding is not None
            for sentence, embedding in zip(sentences, embeddings_list)
        ]
        embeddings_final_list_np = np.squeeze(
            np.stack(
                [embedding for embedding in embeddings_list if embedding is not None]
            )
        )
        return embeddings_final_list_np

    def apply(self, inputs: Sequence[str] | str) -> List["TextEmbedding"]:
        embeddings = self.embed(inputs)
        return embeddings


INSPECT_TEXT_LIST = [
    "Vertexai",
    "AI Platform",
    "Google Cloud",
    "TensorFlow",
    "PyTorch",
    "Keras",
    "Scikit-learn",
    "XGBoost",
    "LightGBM",
    "CatBoost",
    "Hugging Face",
    "Transformers",
    "BERT",
    "GPT-3",
    "OpenAI",
    "Deep Learning",
    "Machine Learning",
    "Natural Language Processing",
    "Computer Vision",
]


def main() -> np.ndarray:
    key_filepath = None
    project_id = "pioneering-flow-199508"
    location = None

    credentials = get_credential(get_key_filepath(key_filepath))
    ic(f"Credentials: {credentials}")

    aiplatform.init(project=project_id, credentials=credentials)
    ic("Initialized AI Platform")

    embedder = TextEmbedder("textembedding-gecko@001")
    inputs = ["Vertexai", "AI Platform"]
    embeddings = embedder.apply(inputs)
    ic(f"Embeddings: {embeddings}")
    embeddings_np = np.array(embeddings)
    ic(f"shape: {embeddings_np.shape}")
    embedding1 = embeddings_np[0].values
    embedding2 = embeddings_np[1].values
    similarity = cosine_similarity([embedding1], [embedding2])
    ic(f"Similarity: {similarity}")

    ic("Embedding texts in batches")
    embeddings_list_np = embedder.embed_texts_in_batches(INSPECT_TEXT_LIST)
    ic(f"Embeddings list_np: {embeddings_list_np}")
    ic(f"shape: {embeddings_list_np.shape}")

    embedder.release()
    return embeddings_list_np


if __name__ == "__main__":
    plot_item_list_np = main()
