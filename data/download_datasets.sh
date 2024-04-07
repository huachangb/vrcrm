#! /bin/bash

# download and untar data in specified directory
function download_from_cornell() {
    wget -O "$1.tgz" "https://www.cs.cornell.edu/~adith/POEM/DATA/$1.tgz"
    mkdir -p $1
    tar -xvzf "$1.tgz" -C $1
    rm "$1.tgz"
}

# download data
download_from_cornell "yeast"
download_from_cornell "scene"
download_from_cornell "tmc2007"
download_from_cornell "lyrl"

echo "Done"
