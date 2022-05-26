from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from TaxiFareModel.gcp import upload_model_to_gcp
from TaxiFareModel.encoders import DistanceTransformer, TimeFeaturesEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from TaxiFareModel.utils import compute_rmse
import numpy as np
from TaxiFareModel.data import get_data, clean_data
from sklearn.model_selection import train_test_split
import pandas as pd

from memoized_property import memoized_property
import mlflow
from mlflow.tracking import MlflowClient

from termcolor import colored
import joblib


# experiment name
EXPERIMENT_NAME = "[UK] [REMOTE] [MICHI] test + v1"

# Indicate mlflow to log to remote server
# mlflow.set_tracking_uri("https://mlflow.lewagon.ai/")

MLFLOW_URI = "https://mlflow.lewagon.ai/"


class Trainer:
    def __init__(self, X, y):
        """
        X: pandas DataFrame
        y: pandas Series
        """
        self.pipeline = None
        self.X = X
        self.y = y

        self.experiment_name = EXPERIMENT_NAME

    def set_pipeline(self):
        """defines the pipeline as a class attribute"""

        # create distance pipeline
        dist_pipe = Pipeline(
            [("dist_trans", DistanceTransformer()), ("stdscaler", StandardScaler())]
        )

        # create time pipeline
        time_pipe = Pipeline(
            [
                ("time_enc", TimeFeaturesEncoder("pickup_datetime")),
                ("ohe", OneHotEncoder(handle_unknown="ignore")),
            ]
        )

        # create preprocessing pipeline
        preproc_pipe = ColumnTransformer(
            [
                (
                    "distance",
                    dist_pipe,
                    [
                        "pickup_latitude",
                        "pickup_longitude",
                        "dropoff_latitude",
                        "dropoff_longitude",
                    ],
                ),
                ("time", time_pipe, ["pickup_datetime"]),
            ],
            remainder="drop",
        )

        # create final pipe
        pipe = Pipeline(
            [("preproc", preproc_pipe), ("linear_model", LinearRegression())]
        )

        self.pipeline = pipe

    def run(self):
        """set and train the pipeline"""
        self.set_pipeline()
        self.pipeline.fit(self.X, self.y)
        print(colored("pipe fitted", "red"))

        # self.mlflow_create_run()
        self.mlflow_log_param("model", str(self.pipeline[-1]))

    def evaluate(self, X_test, y_test):
        """evaluates the pipeline on df_test and return the RMSE"""
        y_predicted = self.pipeline.predict(X_test)

        rmse = compute_rmse(y_predicted, y_test)

        self.mlflow_log_metric("rmse", rmse)

        print(colored(f"RMSE score: {round(rmse,2)}", "magenta"))
        return rmse

    def save_model(self):
        """Save the model into a .joblib format"""
        joblib.dump(self.pipeline, "model.joblib")
        print(colored("model.joblib saved locally", "green"))
        print(colored("model.joblib saved GCP", "green"))
        upload_model_to_gcp()

    @memoized_property
    def mlflow_client(self):
        mlflow.set_tracking_uri(MLFLOW_URI)
        return MlflowClient()

    @memoized_property
    def mlflow_experiment_id(self):
        try:
            print(
                colored(
                    f"experiment URL: https://mlflow.lewagon.ai/#/experiments/{self.mlflow_client.create_experiment(self.experiment_name)}",
                    "yellow",
                )
            )
            return self.mlflow_client.create_experiment(self.experiment_name)
        except BaseException:
            print(
                colored(
                    f"experiment URL: https://mlflow.lewagon.ai/#/experiments/{self.mlflow_client.get_experiment_by_name(self.experiment_name).experiment_id}",
                    "yellow",
                )
            )
            return self.mlflow_client.get_experiment_by_name(
                self.experiment_name
            ).experiment_id

    @memoized_property
    def mlflow_run(self):
        return self.mlflow_client.create_run(self.mlflow_experiment_id)

    def mlflow_log_param(self, key, value):
        self.mlflow_client.log_param(self.mlflow_run.info.run_id, key, value)

    def mlflow_log_metric(self, key, value):
        self.mlflow_client.log_metric(self.mlflow_run.info.run_id, key, value)


if __name__ == "__main__":
    # get data
    df = get_data(aws=False)

    # clean data
    df_clean = clean_data(df)

    y = df["fare_amount"]
    X = df.drop("fare_amount", axis=1)

    # set X and y
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # train
    tr = Trainer(X_train, y_train)
    tr.run()

    # evaluate
    rmse = tr.evaluate(X_test, y_test)

    tr.save_model()
