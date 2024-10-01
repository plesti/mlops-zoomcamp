import requests
from mage_ai.server.constants import VERSION
import pandas as pd
from io import BytesIO
from typing import List

if 'data_loader' not in globals():
    from mage_ai.data_preparation.decorators import data_loader
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test

@data_loader
def ingest_files(**kwargs) -> pd.DataFrame:
    dfs: List[pd.DataFrame] = []

    for year, months in [(2023, (3,))]:
        for i in months:
            response = requests.get(
                f'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year}-{i:02d}.parquet'
            )

            if response.status_code != 200:
                raise Exception(response.text)

            df = pd.read_parquet(BytesIO(response.content))
            dfs.append(df)
    
    df = pd.concat(dfs)

    with open("/home/src/homework_03/metadata.yaml", "r") as f:
        quiz_answers = {
            "Q1": VERSION,
            "Q2": len(f.readlines()),
            "Q3": df.shape[0]
        }

    return df, quiz_answers

@test
def test_output(output, *args) -> None:
    """
    Template code for testing the output of the block.
    """
    assert output is not None, 'The output is undefined'
