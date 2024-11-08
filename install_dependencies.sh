#!/bin/bash

# Update package list
echo "Updating package list..."
sudo apt update

# Install libgl1-mesa-glx and other necessary dependencies
echo "Installing libgl1-mesa-glx and other dependencies..."
sudo apt install -y libgl1-mesa-glx \
                    build-essential \
                    curl \
                    git \
                    cmake \
                    pkg-config \
                    libssl-dev \
                    libcurl4-openssl-dev \
                    libboost-all-dev \
                    python3-dev

# Check if the installation was successful
if [ $? -eq 0 ]; then
    echo "Dependencies installed successfully!"
else
    echo "An error occurred during the installation of dependencies."
    exit 1
fi
