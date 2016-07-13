#!/usr/bin/env bash

cd ~/BookScanning
source virtualenv_python27/bin/activate 

IFS="
"

function generate_directory() {
    if [ -n "$1" ]; then
        echo "Creating $1"
        mkdir "$1"
        echo "Done."
    fi
}

for path in `\find ./pdf/ -maxdepth 1 -name '*.pdf'`; do
    echo "path=$path"
    file=${path##*/}
    echo "file=$file"
    filename=${file%.*}
    echo "filename=$filename"

    dir0="./0/$filename"
    generate_directory $dir0
    echo "Extracting image files"
    python extract_image_from_pdf.py "$path" "$dir0"
    echo "Done."

    dir1="./1/$filename"
    generate_directory "$dir1"
    echo "Reducing noise"
    python reduce_noise.py ./waifu2x/models/vgg_7/art/noise2_model.json "$dir0" "$dir1"
    echo "Done."
done
