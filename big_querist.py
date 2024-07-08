import pandas as pd
from base import Base
from google.cloud import bigquery
from google.oauth2.service_account import Credentials
from loguru import logger
from utils import get_credential, get_key_filepath, get_trainer_script_filepath

key_filepath = None
project_id = "pioneering-flow-199508"


class BigQuerist(Base):
    credentials: Credentials
    project_id: str
    df: pd.DataFrame

    def __init__(self, credentials: Credentials, project_id: str):
        self.credentials = credentials
        self.project_id = project_id

    def apply(self, sql: str) -> pd.DataFrame:
        bq_client = bigquery.Client(
            project=self.project_id, credentials=self.credentials
        )
        job_config = bigquery.QueryJobConfig()
        client_result = bq_client.query(sql, job_config=job_config)
        job_id = client_result.job_id
        df = client_result.result().to_arrow().to_pandas()
        logger.info(f"Finished job_id: {job_id}")
        self.df = df
        return df

    def release(self):
        del self.df


INSPECT_QUERY = """
SELECT
    *
FROM
    `bigquery-public-data.stackoverflow.posts_questions`
LIMIT 3
"""


def main():
    credentials = get_credential(get_key_filepath(key_filepath))
    logger.info(f"Credentials: {credentials}")

    bqr = BigQuerist(credentials, project_id)
    df = bqr.apply(INSPECT_QUERY)
    logger.debug(df.head())
    bqr.release()


if __name__ == "__main__":
    main()
