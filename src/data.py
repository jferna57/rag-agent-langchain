"""
src/data.py

This module defines data structures (dataclasses) and functions for interacting with
Firebase Realtime Database. It provides a way to serialize and save structured
data, including system information, model details, and question-answer pairs.
"""
import dataclasses
import os

from datetime import datetime

from typing import Dict, List

import firebase_admin
from firebase_admin import credentials, db

# Initialize Firebase Admin SDK (only once)
cred = credentials.Certificate(os.getenv("FIREBASE_CREDENTIALS_PATH"))
firebase_admin.initialize_app(cred, {"databaseURL": os.getenv("FIREBASE_URL")})


@dataclasses.dataclass
class SystemInfo:
    """Represents system information."""
    operating_system: str
    version: str
    architecture: str
    processor: str
    physical_cores: int
    logical_cores: int
    ram_gb: float
    disk_space_gb: Dict[str, float]
    python_version: str
    gpu: str
    gpu_count: int


@dataclasses.dataclass
class ModelInfo:
    """Represents information about the model used."""
    model_name: str
    embedding_model: str

@dataclasses.dataclass
class QuestionAnswerPair:
    """Represents a question and its corresponding answer."""
    question: str
    answer: str

@dataclasses.dataclass
class DataPayload:
    """Represents the complete data payload to be saved."""
    server_name: str
    timestamp: str
    server_data: SystemInfo
    performance_data: Dict[str, float]
    model_info: ModelInfo
    questions_and_answers: List[Dict[str, str]]

def serialize_data_payload(data_payload: DataPayload) -> Dict:
    """
    Serializes a DataPayload object into a dictionary suitable for Firebase Realtime Database.
    """
    return {
        "server_name": data_payload.server_name,
        "timestamp": data_payload.timestamp,  # Add the timestamp to the data
        "server_data": {
            "operating_system": data_payload.server_data.operating_system,
            "version": data_payload.server_data.version,
            "architecture": data_payload.server_data.architecture,
            "processor": data_payload.server_data.processor,
            "physical_cores": data_payload.server_data.physical_cores,
            "logical_cores": data_payload.server_data.logical_cores,
            "ram_gb": data_payload.server_data.ram_gb,
            "disk_space_gb": data_payload.server_data.disk_space_gb,
            "python_version": data_payload.server_data.python_version,
            "gpu": data_payload.server_data.gpu,
            "gpu_count": data_payload.server_data.gpu_count,
        },
        "performance_data": data_payload.performance_data.steps_times,
        "model_info": {
            "model_name": data_payload.model_info.model_name,
            "embedding_model": data_payload.model_info.embedding_model
        },
        "questions_and_answers": [
            {"question": qa["question"], "answer": qa["answer"]}  # Access as dictionary keys
            for qa in data_payload.questions_and_answers
        ]
    }


def save_data(data_payload: DataPayload):
    """
    Saves the data payload to Firebase Realtime Database.

    Args:
        data_payload (DataPayload): The data payload object to be saved.
    """
    try:
        now = datetime.now()
        timestamp_str = now.strftime("%Y%m%d%H%M%S%f") 
        timestamp_human = now.strftime("%Y-%m-%d %H:%M:%S")
        data_payload.timestamp = timestamp_human

        # Serialize the DataPayload object, including the timestamp
        data = serialize_data_payload(data_payload)

        # Get a reference to the database, creating a new node with the timestamp in the key
        ref = db.reference(f"/{data_payload.server_name}/{timestamp_str}")

        # Push the data to the database
        ref.set(data)

        print(f"Data sent successfully to Firebase at /{data_payload.server_name}")

    except (Exception) as e:
        print(f"An error occurred while saving data to Firebase: {e}")
