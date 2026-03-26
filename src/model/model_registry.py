## Importing necessary libraries:
import os

## Data manipulation
import numpy as np

## Experiment tracking
import mlflow

## File saving
import json

from config import aws_uri
from utils import setup_logger

mlflow.set_tracking_uri(aws_uri)
logger = setup_logger("model_registry", "registration_error.log")

def load_model_info(file_path: str)-> dict:
    '''Load run_id and location of model in the S3 bucket'''
    try:

        with open(file_path, "r") as f:
            model_info = json.load(f)
        logger.debug("Model_info loaded")
        return model_info
    except FileNotFoundError:
        logger.error("File not found: %s", file_path)
        raise
    except Exception as e:
        logger.error("Unexpect error: %s", e)
        raise

def register_model(VAE_name: str, ISO_name: str, model_info: dict):
    '''Register model to MLflow'''
    try:
        
        VAE_uri = model_info["VAE_path"]
        ISO_uri = model_info["ISO_path"]

        VAE_version = mlflow.register_model(VAE_uri, VAE_name)
        ISO_version = mlflow.register_model(ISO_uri, ISO_name)

        # Moving model to staging stage

        client = mlflow.tracking.MlflowClient()
        client.transition_model_version_stage(
            name = VAE_name,
            version = VAE_version.version,
            stage = "Staging"
        )

        client.transition_model_version_stage(
            name = ISO_name,
            version = ISO_version.version,
            stage = "Staging"
        )

        logger.debug(f"VAE Version: {VAE_version.version} and Isolation Forest version: {ISO_version.version} registered and moved to staging.")

    except Exception as e:
        logger.error("Model registry failed: %s",e)
        raise

def main():

    try:

        model_info_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),"../Metadata/experiment_info.json")

        model_info = load_model_info(file_path= model_info_path)

        VAE_name = "VAE_anomaly_detection"
        ISO_name = "Isolation_Forest_anomaly_detection"

        register_model(VAE_name=VAE_name, ISO_name=ISO_name, model_info=model_info)

    except Exception as e:
        logger.error("Failed to log model registry: %s",e)
        raise

if __name__ == "__main__":
    main()






