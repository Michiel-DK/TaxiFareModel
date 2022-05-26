from google.cloud import storage
import joblib
import os

STORAGE_LOCATION = "models/TaxiFareModel/model.joblib"
BUCKET_NAME = "wagon-data-867-dk"


def upload_model_to_gcp():

    client = storage.Client()

    bucket = client.bucket(BUCKET_NAME)

    blob = bucket.blob(STORAGE_LOCATION)

    blob.upload_from_filename("model.joblib")


def download_model(model_directory="TaxiFareModel", bucket=BUCKET_NAME, rm=True):
    client = storage.Client().bucket(bucket)

    storage_location = "models/{}/{}".format(model_directory, "model.joblib")
    blob = client.blob(storage_location)
    blob.download_to_filename("model.joblib")
    print("=> pipeline downloaded from storage")
    model = joblib.load("model.joblib")
    if rm:
        os.remove("model.joblib")
    return model
