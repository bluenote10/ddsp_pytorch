#!/bin/bash

echo "Updating drive before modifying..."
drive_pull

input_dir="${HOME}/gdrive/colab/ddsp_pytorch/input"
tar_file="$input_dir/ddsp.tar.gz"

echo "Packaging to: $tar_file"
mkdir -p "$input_dir"
git ls-files | tar Tzcf - "$tar_file"

echo "Pushing to drive..."
drive_push
