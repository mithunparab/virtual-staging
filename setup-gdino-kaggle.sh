#!/bin/bash
echo "--- Setting up project structure and dependencies ---"

cd /kaggle/working/

echo "Cloning GroundingDINO repository..."
git clone -q https://github.com/IDEA-Research/GroundingDINO.git


echo "Creating weights directory..."
mkdir -p weights

echo "Downloading GroundingDINO weights..."
wget -q -O weights/groundingdino_swint_ogc.pth https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth


pip install -q groundingdino-py ultralytics==8.2.2 addict yapf

export PYTHONPATH="/kaggle/working/GroundingDINO:$PYTHONPATH"

echo "--- Setup complete ---"