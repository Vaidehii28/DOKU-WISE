import json
import os
import logging
import torch
from google.cloud import vision
from google.oauth2 import service_account
from utilities.constants import service_account_path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

credentials = service_account.Credentials.from_service_account_file(service_account_path)
client = vision.ImageAnnotatorClient(credentials=credentials)

class FoldersConfiguration:
    DATASET_PATH = os.path.join("data", "dataset")
    DUMPS_PATH = os.path.join("data", "dumps")
    STATUS_PATH = os.path.join("data", "status")
    DOWNLOADS_PATH = os.path.join("data", "downloads")


class FileNamesConfiguration:
    DATASET_DUMP = "dataset_details_dump.csv"
    DETAILS_DUMP = "document_details_dump.csv"
    INFO = "info.json"
    SUCCESSFULLY_PROCESSED_APPLICATION_ID = "successfully_processed_application_id.txt"
    SUCCESSFULLY_DOWNLOADED_STATEMENTS = "successfully_downloaded_statements.txt"
    
    
class FilePathsConfiguration:

    DATASET_DUMP_PATH = os.path.join(
        FoldersConfiguration.DATASET_PATH,
        FileNamesConfiguration.DATASET_DUMP
    )
    
    DETAILS_DUMP_PATH = os.path.join(
        FoldersConfiguration.DUMPS_PATH,
        FileNamesConfiguration.DETAILS_DUMP
    )

    SUCCESSFULLY_DOWNLOADED_STATEMENTS_STATUS_PATH = os.path.join(
        FoldersConfiguration.STATUS_PATH,
        FileNamesConfiguration.SUCCESSFULLY_DOWNLOADED_STATEMENTS
    )
