import pandas as pd
import batch
from datetime import datetime

def dt(hour, minute, second=0):
    return datetime(2023, 1, 1, hour, minute, second)


def test_prepare_data():
    input_data = [
        (None, None, dt(1, 1), dt(1, 10)),
        (1, 1, dt(1, 2), dt(1, 10)),
        (1, None, dt(1, 2, 0), dt(1, 2, 59)),
        (3, 4, dt(1, 2, 0), dt(2, 2, 1)),
    ]

    columns = ['PULocationID', 'DOLocationID', 'tpep_pickup_datetime', 'tpep_dropoff_datetime']
    df = pd.DataFrame(input_data, columns=columns)

    df = batch.prepare_data(df, [])

    # Duration column should be added by the prepare_data(..) function
    expected_columns = columns + ["duration"]
    expected = pd.DataFrame([
        (None, None, dt(1, 1), dt(1, 10), 9.0),
        (1, 1, dt(1, 2), dt(1, 10), 8.0),
    ], columns=expected_columns)

    # Check value and dtype differences of the input and expected dataframe
    pd.testing.assert_frame_equal(df, expected)