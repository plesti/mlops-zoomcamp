pip install pipenv

# Install deployment and development python packages
pipenv install --dev

# Create "nyc-duration" bucket (documentation https://docs.localstack.cloud/user-guide/integrations/aws-cli/#configuring-an-endpoint-url)
pipenv run aws --endpoint-url=$S3_ENDPOINT_URL s3 mb s3://nyc-duration
pipenv run aws --endpoint-url=$S3_ENDPOINT_URL s3 ls
# Alternatively using localstack -> https://docs.localstack.cloud/user-guide/aws/s3/#create-an-s3-bucket
# awslocal s3api list-buckets
# awslocal s3api create-bucket --bucket sample-bucket

# Run homework
# Pytest saves dummy dataframe to S3 storage and batch.py reads it
export INPUT_FILE_PATTERN=s3://nyc-duration/taxi_type=fhv/year={year:04d}/month={month:02d}/yellow_tripdata.parquet
pipenv run pytest
pipenv run python batch.py 2023 01

# Keep container running for a day
echo Sleeping...
sleep 1d