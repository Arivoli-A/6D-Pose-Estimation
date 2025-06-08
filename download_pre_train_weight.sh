#!/bin/bash

# Define the URL and the target directory
url="https://cns-data.aau.at/ai-modules-poet/poet_ycbv_yolo.pth"
target_dir="./pre_train"

# Create the target directory if it doesn't exist
mkdir -p "$target_dir"

# Download the file using wget
file_name=$(basename "$url") # poet_ycbv_yolo.pth
wget "$url" -P "$target_dir"

# Save the name of the downloaded file
echo "Downloaded file: $file_name"

# Append the file name into a .txt file
echo "$file_name" >> "$target_dir/pre_train_weight_name.txt"

# Optional: Store the file name in a variable for later use
saved_file="$target_dir/$file_name"
echo "File saved to: $saved_file"
