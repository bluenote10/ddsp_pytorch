#!/bin/bash

input_dir="${HOME}/gdrive/colab/ddsp_pytorch/input"

mkdir -p "$input_dir"

tar_file="$input_dir/ddsp.tar.gz"

echo "Packaging to: $tar_file"
git ls-files | tar Tzcf - "$tar_file"

drive_push
