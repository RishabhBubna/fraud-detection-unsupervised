import os

# Base directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Raw data paths — override with env variables if needed
RAW_TRANSACTION_PATH  = os.getenv("RAW_TRANSACTION_PATH",  os.path.join(BASE_DIR, "ieee-fraud-detection", "train_transaction.csv"))
RAW_IDENTITY_PATH     = os.getenv("RAW_IDENTITY_PATH",     os.path.join(BASE_DIR, "ieee-fraud-detection", "train_identity.csv"))
TEST_TRANSACTION_PATH = os.getenv("TEST_TRANSACTION_PATH", os.path.join(BASE_DIR, "ieee-fraud-detection", "test_transaction.csv"))
TEST_IDENTITY_PATH    = os.getenv("TEST_IDENTITY_PATH",    os.path.join(BASE_DIR, "ieee-fraud-detection", "test_identity.csv"))

# MLflow tracking URI
aws_uri = os.getenv("MLFLOW_TRACKING_URI")