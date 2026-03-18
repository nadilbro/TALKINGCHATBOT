#!/bin/bash
set -e

pip install -r requirements.txt

# Install Rhubarb
wget -q https://github.com/DanielSWolf/rhubarb-lip-sync/releases/download/v1.13.0/rhubarb-lip-sync-1.13.0-linux.zip
unzip -q rhubarb-lip-sync-1.13.0-linux.zip

# Binary is inside a subfolder, not the root
mv rhubarb-lip-sync-1.13.0-linux/rhubarb /usr/local/bin/rhubarb
chmod +x /usr/local/bin/rhubarb

echo "==> Rhubarb installed successfully"
rhubarb --version