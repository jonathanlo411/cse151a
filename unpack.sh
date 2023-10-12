#!/bin/bash

# Specify the directory containing .gz files
directory_path="data/"

# Loop through the directory and unzip .gz files
for gz_file in "$directory_path"*.gz; do
    if [ -e "$gz_file" ]; then
        # Extract the file name without the .gz extension
        filename=$(basename "$gz_file" .gz)
        
        # Unzip the file
        gunzip -c "$gz_file" > "$directory_path$filename"
        
        echo "Uncompressed '$gz_file' to '$directory_path$filename'"
    fi
done

echo "All .gz files have been uncompressed."
