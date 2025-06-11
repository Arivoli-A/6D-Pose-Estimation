#!/bin/bash

# Usage:
# ./download_and_unzip.sh <file_id> <output_folder> <conda_env_name>

# Defaults
DEFAULT_FILE_ID="1AYpqt55xax4nI_70a-eha-3hy1KXuEng" 
DEFAULT_BASE_FOLDER_NAME="./dataset/HOPE"
DEFAULT_CONDA_ENV_NAME="pose_estimation"

# Input arguments with defaults
FILE_ID="${1:-$DEFAULT_FILE_ID}"
BASE_FOLDER_NAME="${2:-$DEFAULT_BASE_FOLDER_NAME}"
CONDA_ENV_NAME="${3:-$DEFAULT_CONDA_ENV_NAME}"


# Generate timestamped output folder
TIMESTAMP=$(date +"%Y%m%d") #_%H%M%S
OUTPUT_FOLDER="${BASE_FOLDER_NAME}_${TIMESTAMP}"

# Activate Conda environment
echo "[INFO] Activating conda environment: $CONDA_ENV_NAME"
source activate
conda activate "$CONDA_ENV_NAME"

# Check if gdown is installed
if ! command -v gdown; then
    echo "[INFO] gdown not found. Installing..."
    pip install gdown
else
    echo "[INFO] gdown is already installed."
fi

# Create output folder if it doesn't exist
mkdir -p "$OUTPUT_FOLDER"

# Download the file using gdown
DOWNLOAD_URL="https://drive.google.com/uc?id=${FILE_ID}"
echo "[INFO] Downloading file from Google Drive..."
gdown --output "${DEFAULT_BASE_FOLDER_NAME}/" "$DOWNLOAD_URL"

# Unzip the downloaded file
#echo "[INFO] Unzipping file to $OUTPUT_FOLDER"
#unzip -o "${DEFAULT_BASE_FOLDER_NAME}/valid_npz_outputs_split.zip" -d "$OUTPUT_FOLDER"  

rm -rf $DEFAULT_BASE_FOLDER_NAME

# Deactivate Conda environment
echo "[INFO] Deactivating conda environment"
conda deactivate

echo "[INFO] Done."
