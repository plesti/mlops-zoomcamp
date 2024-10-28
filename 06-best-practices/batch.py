#!/usr/bin/env python
# coding: utf-8
import os
import sys
import pickle
import pandas as pd
import re
import boto3

S3_DEFAULT_BUCKET_NAME = 'nyc-duration'

def get_s3_upload_options(filename):
    if filename.startswith("s3://"):
        S3_ENDPOINT_URL = os.getenv("S3_ENDPOINT_URL")
        if not S3_ENDPOINT_URL:
            raise ValueError(f"Please set S3_ENDPOINT_URL environment variable to use S3 storage access: {filename}")

        return {
            'client_kwargs': {
                'endpoint_url': S3_ENDPOINT_URL
            }
        }
    return None

def get_s3_file_size(s3path):
    try:
        bucket, s3_fp = re.search("s3://([^/]+)/(.+)", s3path).groups()
    except:
        raise RuntimeError(f"Failed regex for: {s3path}")
    connection = boto3.client(service_name="s3", endpoint_url=os.getenv("S3_ENDPOINT_URL"))
    resp = connection.list_objects(Bucket=bucket, Prefix=s3_fp)
    try:
        return resp['Contents'][0]['Size']
    except KeyError:
        print(f"Failed to find file '{s3_fp}' in s3 bucket '{bucket}'")
        raise

def read_data(filename):
    options = get_s3_upload_options(filename)
    df = pd.read_parquet(filename, storage_options=options)
    return df

def save_data(dataframe : pd.DataFrame, filename):
    options = get_s3_upload_options(filename)

    dataframe.to_parquet(
        filename,
        engine='pyarrow',
        compression=None,
        index=False,
        storage_options=options
    )

def prepare_data(df, categorical):
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    return df

def get_input_path(year, month):
    default_input_pattern = 'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year:04d}-{month:02d}.parquet'
    input_pattern = os.getenv('INPUT_FILE_PATTERN', default_input_pattern)
    return input_pattern.format(year=year, month=month)

def get_output_path(year, month):
    default_output_pattern = f's3://nyc-duration/taxi_type=fhv/year={year:04d}/month={month:02d}/predictions.parquet'
    output_pattern = os.getenv('OUTPUT_FILE_PATTERN', default_output_pattern)
    return output_pattern.format(year=year, month=month)

def main(year, month):
    input_file = get_input_path(year, month)
    output_file = get_output_path(year, month)
    categorical = ['PULocationID', 'DOLocationID']

    with open('model.bin', 'rb') as f_in:
        dv, lr = pickle.load(f_in)

    df = read_data(input_file)
    df = prepare_data(df, categorical)

    df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')

    dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(dicts)
    y_pred = lr.predict(X_val)

    print('predicted mean duration:', y_pred.mean())

    df_result = pd.DataFrame()
    df_result['ride_id'] = df['ride_id']
    df_result['predicted_duration'] = y_pred

    save_data(df_result, output_file)

if __name__ == '__main__':
    year = int(sys.argv[1])
    month = int(sys.argv[2])

    main(year, month)

    # Q1. Refactoring
    print("Q1. How does the if statement that we use for this looks like?\n  Answer: \"if __name__ == '__main__':\"")

    # Q2. Installing pytest
    print(f"Q2. What should be the other file?\n  Answer: __init__.py")

    # Q3. Writing first unit test
    # The expected dataframe is defined in tests.test_batch.test_prepare_data function
    print(f"Q3. How many rows should be there in the expected dataframe?\n  Answer: 2")  # 2

    # Q4. Mocking S3 with Localstack
    print(f"Q4. What option do we need to use for such purposes?\n  Answer: --endpoint-url")  # --endpoint-url

    # Q5. Creating test data
    # Alternative to list files from cli:
    #   aws --endpoint-url=$S3_ENDPOINT_URL s3 ls s3://nyc-duration/taxi_type=fhv/year=2023/month=01/
    try:
        s3path = get_input_path(year, month)
        file_size = get_s3_file_size(s3path)
        print(f"Q5. What's the size of the file?\n  Answer: {file_size}")  # 3620
    except Exception as e:
        print(f"Q5. What's the size of the file?\n Failed with exception {e}. Please run the following command "
              f"before execution:\n  pipenv run python -m unittest -v tests.test_batch tests.integration_test")


    # Q6. Finish the integration test
    output_file = get_output_path(year, month)
    df = read_data(output_file)
    print(f"Q6. What's the sum of predicted durations for the test dataframe?\n"
          f"  Answer: {df.predicted_duration.sum()}")  # 36.28
