#!/bin/bash
set -e  # stop if anything fails

# Install Python deps
pip install -r requirements.txt

# Install Rhubarb binary
wget -q https://github.com/DanielSWolf/rhubarb-lip-sync/releases/download/v1.13.0/rhubarb-lip-sync-1.13.0-linux.zip
unzip -q rhubarb-lip-sync-1.13.0-linux.zip
mv rhubarb /usr/local/bin/rhubarb
chmod +x /usr/local/bin/rhubarb
```

---

## 3. Update your Render build command

In your Render dashboard go to your service → **Settings** → **Build Command** and change it from:
```
pip install -r requirements.txt
```
to:
```
bash build.sh