#!/bin/bash

# Define source and destination paths
SOURCE_DIR="./Deformable-DETR/models/ops/build/lib.linux-x86_64-cpython-311"
DEST_DIR="./src/model/deformable_attention"

echo "Starting directory copy process..."
echo "Source: $SOURCE_DIR"
echo "Destination: $DEST_DIR"

# Check if the source directory exists
if [ ! -d "$SOURCE_DIR" ]; then
  echo "Error: Source directory not found at '$SOURCE_DIR'."
  echo "Please ensure you are running this script from the correct location or adjust the SOURCE_DIR variable."
  exit 1
fi

# Create the destination directory if it doesn't exist
if [ ! -d "$DEST_DIR" ]; then
  echo "Destination directory '$DEST_DIR' does not exist. Creating it..."
  mkdir -p "$DEST_DIR"
  if [ $? -ne 0 ]; then
    echo "Error: Failed to create destination directory '$DEST_DIR'."
    exit 1
  fi
fi

# Copy contents from source to destination
# -v: verbose, shows what's being copied
# -r: recursive, copies directories and their contents
echo "Copying directories from '$SOURCE_DIR' to '$DEST_DIR'..."
cp -vr "$SOURCE_DIR"/* "$DEST_DIR"/

if [ $? -eq 0 ]; then
  echo "Successfully copied contents to '$DEST_DIR'."
else
  echo "An error occurred during the copy process."
  exit 1
fi

echo "Script finished."