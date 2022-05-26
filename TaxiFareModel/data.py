import pandas as pd
from termcolor import colored
from google.cloud import storage
from TaxiFareModel.utils import df_optimized

AWS_BUCKET_PATH = "s3://wagon-public-datasets/taxi-fare-train.csv"
BUCKET_TRAIN_DATA_PATH = "data/train_1k.csv"
BUCKET_NAME = "wagon-data-867-dk"


def get_data(nrows=10_000, aws=True):
    """returns a DataFrame with nrows from s3 bucket"""
    if aws:
        df = pd.read_csv(AWS_BUCKET_PATH, nrows=nrows)
        print(colored(f"data retrieved from {AWS_BUCKET_PATH}", "blue"))
    else:
        df = pd.read_csv(f"gs://{BUCKET_NAME}/{BUCKET_TRAIN_DATA_PATH}", nrows=1000)
        print(colored(f"data retrieved from {BUCKET_NAME}", "blue"))
    return df_optimized(df)


def clean_data(df, test=False):
    df = df.dropna(how="any", axis="rows")
    df = df[(df.dropoff_latitude != 0) | (df.dropoff_longitude != 0)]
    df = df[(df.pickup_latitude != 0) | (df.pickup_longitude != 0)]
    if "fare_amount" in list(df):
        df = df[df.fare_amount.between(0, 4000)]
    df = df[df.passenger_count < 8]
    df = df[df.passenger_count >= 0]
    df = df[df["pickup_latitude"].between(left=40, right=42)]
    df = df[df["pickup_longitude"].between(left=-74.3, right=-72.9)]
    df = df[df["dropoff_latitude"].between(left=40, right=42)]
    df = df[df["dropoff_longitude"].between(left=-74, right=-72.9)]
    return df_optimized(df)


if __name__ == "__main__":
    df = get_data()
