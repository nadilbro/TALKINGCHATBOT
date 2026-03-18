#!/bin/bash
set -e

pip install -r requirements.txt

# Just check what's available
which espeak || echo "espeak not found"
which espeak-ng || echo "espeak-ng not found"

# Rhubarb
wget -q https://your-url/rhubarb-lip-sync-1.13.0-linux.zip
unzip -q rhubarb-lip-sync-1.13.0-linux.zip
cp Rhubarb-Lip-Sync-1.13.0-Linux/rhubarb ./rhubarb
chmod +x ./rhubarb