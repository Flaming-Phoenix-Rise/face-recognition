#!/bin/bash

# Example script to install dependencies on a Linux-based system
echo "Updating package list..."
apt-get update

echo "Installing libgl1-mesa-glx and other dependencies..."
apt-get install -y libgl1-mesa-glx \
                  build-essential \
                  curl \
                  git \
                  cmake \
                  pkg-config \
                  libssl-dev \
                  libcurl4-openssl-dev \
                  libboost-all-dev \
                  python3-dev

# Check if installation was successful
if [ $? -eq 0 ]; then
    echo "Dependencies installed successfully!"
else
    echo "An error occurred during the installation of dependencies."
    exit 1
fi
