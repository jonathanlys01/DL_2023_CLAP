#!/bin/bash

# Create a directory for the datasets (if it does not exist)
mkdir -p downloads
cd downloads

# Download the datasets if they are not already downloaded
counter=0

# ESC-50
if [ ! -d "ESC-50-master" ]; then
    echo "ESC-50-master not found, downloading..."
    wget https://github.com/karoldvl/ESC-50/archive/master.zip
    echo "Unzipping..."
    unzip -q master.zip
    rm master.zip
    ((counter++))
else
    echo "ESC-50-master found, skipping..."
fi

# UrbanSound8K
if [ ! -d "UrbanSound8K" ]; then
    echo "UrbanSound8K not found, downloading..."
    wget https://zenodo.org/record/1203745/files/UrbanSound8K.tar.gz
    echo "Unzipping..."
    tar -xzf UrbanSound8K.tar.gz
    rm UrbanSound8K.tar.gz
    ((counter++))
else
    echo "UrbanSound8K found, skipping..."
fi

# FMA Small
if [ ! -d "fma_small" ]; then
    echo "fma_small not found, downloading..."
    curl -O https://os.unil.cloud.switch.ch/fma/fma_small.zip
    echo "ade154f733639d52e35e32f5593efe5be76c6d70  fma_small.zip"    | sha1sum -c -
    echo "Unzipping..."
    unzip -q fma_small.zip
    rm fma_small.zip
    ((counter++))
else
    echo "fma_small found, skipping..."
fi

# FMA Metadata
if [ ! -d "fma_metadata" ]; then
    echo "fma_metadata not found, downloading..."
    wget https://os.unil.cloud.switch.ch/fma/fma_metadata.zip
    echo "Unzipping..."
    unzip -q fma_metadata.zip
    rm fma_metadata.zip
    ((counter++))
else
    echo "fma_metadata found, skipping..."
fi

echo "Downloaded $c datasets."

echo "Size of the datasets:"
du -sh ./*

echo "chmod 777 -R downloads ?"
read answer
if [ "$answer" != "${answer#[Yy]}" ] ;then
    chmod 777 -R ../downloads
fi
cd ..
echo "Done !"

