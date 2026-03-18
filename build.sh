#!/bin/bash
set -e

pip install -r requirements.txt

# Install espeak (required by Rhubarb's phonetic recognizer)
apt-get install -y espeak

# Install Rhubarb
wget -q https://your-r2-or-github-url/rhubarb-lip-sync-1.13.0-linux.zip
unzip -q rhubarb-lip-sync-1.13.0-linux.zip
cp Rhubarb-Lip-Sync-1.13.0-Linux/rhubarb ./rhubarb
chmod +x ./rhubarb

echo "==> Rhubarb installed:"
./rhubarb --version