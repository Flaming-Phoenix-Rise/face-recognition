#!/bin/bash

# Update the package list
echo "Updating package list..."
sudo apt-get update

# Install required libraries
echo "Installing libgl1-mesa-glx..."
sudo apt-get install -y libgl1-mesa-glx

# You can add more dependencies here if needed
echo "All required dependencies installed."
