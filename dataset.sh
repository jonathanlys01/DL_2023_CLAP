#!/bin/bash

# Create a directory for the datasets (if it does not exist)
mkdir -p downloads
cd downloads

# Download the datasets if they are not already downloaded
# Count the number of datasets downloaded
$c = 0

# ESC-50
if [ ! -d "ESC-50-master" ]; then
    wget https://github.com/karoldvl/ESC-50/archive/master.zip
    unzip -q master.zip
    rm master.zip
    $c = $c + 1
fi

# UrbanSound8K
if [ ! -d "UrbanSound8K" ]; then
    wget https://zenodo.org/record/1203745/files/UrbanSound8K.tar.gz
    tar -xzf UrbanSound8K.tar.gz
    rm UrbanSound8K.tar.gz
    $c = $c + 1
fi

# FMA Small
if [ ! -d "fma_small" ]; then
    wget https://os.unil.cloud.switch.ch/fma/fma_small.zip
    unzip -q fma_small.zip
    rm fma_small.zip
    $c = $c + 1
fi

# FMA Metadata
if [ ! -d "fma_metadata" ]; then
    wget https://os.unil.cloud.switch.ch/fma/fma_metadata.zip
    unzip -q fma_metadata.zip
    rm fma_metadata.zip
    $c = $c + 1
fi

echo "Downloaded $c datasets."

echo "Size of the datasets:"
du -sh ./*

echo "chmod 777 -R downloads ?"
read answer
if [ "$answer" != "${answer#[Yy]}" ] ;then
    chmod 777 -R downloads
fi

echo "Done !"

