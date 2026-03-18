#!/bin/bash
set -e

pip install -r requirements.txt

# Install espeak statically (no apt needed)
wget -q https://github.com/espeak-ng/espeak-ng/releases/download/1.51/espeak-ng-1.51-linux-x64.zip
unzip -q espeak-ng-1.51-linux-x64.zip
cp espeak-ng ./espeak-ng
chmod +x ./espeak-ng

# Install Rhubarb
wget -q https://your-url/rhubarb-lip-sync-1.13.0-linux.zip
unzip -q rhubarb-lip-sync-1.13.0-linux.zip
cp Rhubarb-Lip-Sync-1.13.0-Linux/rhubarb ./rhubarb
chmod +x ./rhubarb

echo "==> Done"
./rhubarb --version