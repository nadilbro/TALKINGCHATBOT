#!/bin/bash
set -e

pip install -r requirements.txt

wget -q https://github.com/DanielSWolf/rhubarb-lip-sync/releases/download/v1.13.0/rhubarb-lip-sync-1.13.0-linux.zip
unzip -q rhubarb-lip-sync-1.13.0-linux.zip

cp Rhubarb-Lip-Sync-1.13.0-Linux/rhubarb ./rhubarb
chmod +x ./rhubarb

echo "==> Rhubarb installed at:"
ls -la ./rhubarb