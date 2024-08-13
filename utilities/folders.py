import os

from utilities.config import logger

def create_folder_if_not_exists(folder_path: str) -> None:
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        logger.info(f"Created folder: {folder_path}")
    else:
        logger.info(f"Folder already exists: {folder_path}")

def delete_folder_if_exists(folder_path):
    if os.path.exists(folder_path):
        os.remove(path=folder_path)
        logger.info(f"deleted folder {folder_path}")
