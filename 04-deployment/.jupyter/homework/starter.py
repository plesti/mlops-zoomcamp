#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import pickle
import pandas as pd
import subprocess

try:
    # Check if script is running from ipython notebook
    get_ipython()
    MAKE_PACKAGE = True
except NameError:
    # Script is running from console
    MAKE_PACKAGE = False

if MAKE_PACKAGE:
    get_ipython().system('pip freeze | grep scikit-learn')
    get_ipython().system('python -V')
    # sklearn version is prefered to be matching with the pickled model.bin file's sklearn version
    get_ipython().system('pip install -U scikit-learn==1.5.0')
    # Docker is needed to build image in Q6 answer.
    # I have mounted my host's /var/run/docker.sock to the jupyter environment to access docker service itself
    get_ipython().system('apt-get update')
    get_ipython().system('apt-get install -y docker.io')


# ## Q5.1: Parametrize the script (Output is after Q4 Answer)

# In[2]:


try:
    year, month = int(os.environ["YEAR"]), int(os.environ["MONTH"])
except KeyError:
    year, month = 2023, 3
    print(f"Missing year and month input arguments. Using default values (year, month): {year, month}")


# ## Predict values

# In[3]:


with open('model.bin', 'rb') as f_in:
    dv, model = pickle.load(f_in)


# In[4]:


categorical = ['PULocationID', 'DOLocationID']

def read_data(filename):
    print(f"Loading parquet file: {DATA_URL}")
    df = pd.read_parquet(filename)
    
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')

    return df


# In[ ]:


# Download dataset
DATA_URL = f'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year:04d}-{month:02d}.parquet'
df = read_data(DATA_URL)

dicts = df[categorical].to_dict(orient='records')
X_val = dv.transform(dicts)
y_pred = model.predict(X_val)


# ## Q1: What's the standard deviation of the predicted duration for this dataset?

# In[ ]:


if MAKE_PACKAGE:
    print(f"Q1: What's the standard deviation of the predicted duration for this dataset?\n  Answer: {y_pred.std()}")  # 6.247


# ## Q2: Preparing the output

# In[ ]:


if MAKE_PACKAGE:
    output_file = f"scoring_data_{year:04d}-{month:02d}.parquet"
    
    df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')
    df_result = pd.DataFrame()
    df_result['prediction'] = y_pred
    df_result['ride_id'] = df['ride_id']
    df_result.to_parquet(
        output_file,
        engine='pyarrow',
        compression=None,
        index=False
    )
    print(f"Q2: What's the size of the output file?\n  Answer: Size of {output_file} is {os.stat(output_file).st_size/(1024*1024):.0f}MB")  # 64MB


# ## Q3: Creating the scoring script

# In[ ]:


if MAKE_PACKAGE:
    print("Q3: Which command you need to execute to turn the notebook into a script? Answer: 'jupyter nbconvert --to script starter.ipynb'")
    print(subprocess.getoutput("jupyter nbconvert --to script starter.ipynb"))


# ## Q4: Virtual environment

# In[ ]:


if MAKE_PACKAGE:
    print(subprocess.getoutput("pip install pipenv"))
    print(subprocess.getoutput("python3 -m pipenv install pandas scikit-learn==1.5.0 pyarrow --python=3.10"))


# In[ ]:


if MAKE_PACKAGE:
    import json
    with open("Pipfile.lock", "rb") as f:
        hash = json.load(f)["default"]["scikit-learn"]["hashes"][0]
        print(f"Q4: What's the first hash for the Scikit-Learn dependency?\n  Answer: {hash}")  # sha256:057b991ac64b3e75c9c04b5f9395eaf19a6179244c089afdebaad98264bff37c


# ## Q5.2 Parametrize the script (Answer)

# In[ ]:


if MAKE_PACKAGE:
    print(subprocess.getoutput('YEAR=2023 MONTH=04 Q5=somevalue python3 starter.py'))
if os.environ.get("Q5", False):
    print(f"Q5: What's the mean predicted duration?\n  Answer: {y_pred.mean()}")  # 14.29


# ## Q6: Docker container

# In[ ]:


if MAKE_PACKAGE:
    print(subprocess.getoutput("docker rmi homework4"))
    print(subprocess.getoutput("docker build -t homework4 ."))


# In[ ]:


if MAKE_PACKAGE:
    print(subprocess.getoutput('docker run -i --rm -e YEAR=2023 -e MONTH=05 -e Q6=somevalue homework4'))
if os.environ.get("Q6", False):
    print(f"Q6: What's the mean predicted duration?\n  Answer: {y_pred.mean()}")  # 0.1917

