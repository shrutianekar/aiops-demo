import os
import json
import shutil
import hashlib
from datetime import datetime

# Define dataset storage path
DATASET_DIR = "datasets"
VERSION_METADATA_FILE = os.path.join(DATASET_DIR, "dataset_versions.json")

# Ensure dataset directory exists
os.makedirs(DATASET_DIR, exist_ok=True)

def hash_dataset(file_path):
    """Generate hash for a dataset file to track changes."""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(8192):
            hasher.update(chunk)
    return hasher.hexdigest()

def save_dataset_version(source_path, version_name=None):
    """Save a new version of the dataset with metadata tracking."""
    if not os.path.exists(source_path):
        raise FileNotFoundError(f"Dataset file {source_path} not found.")

    dataset_hash = hash_dataset(source_path)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    version_name = version_name or f"version_{timestamp}"

    version_dir = os.path.join(DATASET_DIR, version_name)
    os.makedirs(version_dir, exist_ok=True)
    
    # Copy dataset
    dest_path = os.path.join(version_dir, os.path.basename(source_path))
    shutil.copy2(source_path, dest_path)

    # Update metadata
    version_info = {
        "version": version_name,
        "file_name": os.path.basename(source_path),
        "path": dest_path,
        "timestamp": timestamp,
        "hash": dataset_hash
    }

    # Load existing metadata
    if os.path.exists(VERSION_METADATA_FILE):
        with open(VERSION_METADATA_FILE, "r") as f:
            metadata = json.load(f)
    else:
        metadata = []

    metadata.append(version_info)

    # Save updated metadata
    with open(VERSION_METADATA_FILE, "w") as f:
        json.dump(metadata, f, indent=4)

    print(f"Dataset version '{version_name}' saved successfully.")
    return version_info

def list_dataset_versions():
    """List all saved dataset versions."""
    if not os.path.exists(VERSION_METADATA_FILE):
        print("No dataset versions found.")
        return []
    
    with open(VERSION_METADATA_FILE, "r") as f:
        metadata = json.load(f)

    return metadata

if __name__ == "__main__":
    # Example usage
    dataset_path = "sample_dataset.csv"  # Replace with actual dataset path
    save_dataset_version(dataset_path)
    print(list_dataset_versions())