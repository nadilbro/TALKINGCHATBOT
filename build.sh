#!/bin/bash
set -e

pip install -r requirements.txt

wget -q https://github.com/DanielSWolf/rhubarb-lip-sync/releases/download/v1.13.0/rhubarb-lip-sync-1.13.0-linux.zip
unzip -q rhubarb-lip-sync-1.13.0-linux.zip

mv Rhubarb-Lip-Sync-1.13.0-Linux/rhubarb /usr/local/bin/rhubarb
chmod +x /usr/local/bin/rhubarb

echo "==> Rhubarb installed at:"
which rhubarb
rhubarb --version