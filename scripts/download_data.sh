#!/usr/bin/env bash
# Downloads and extracts the Kaggle "Chest X-Ray Images (Pneumonia)" dataset
# into data/chest_xray. Requires the `kaggle` CLI to be installed and
# configured (see the README's "Getting the data" section).
set -euo pipefail

DEST_DIR="${1:-data}"

mkdir -p "$DEST_DIR"
cd "$DEST_DIR"

echo "Downloading paultimothymooney/chest-xray-pneumonia from Kaggle..."
kaggle datasets download -d paultimothymooney/chest-xray-pneumonia

echo "Extracting..."
unzip -q chest-xray-pneumonia.zip
rm chest-xray-pneumonia.zip
rm -rf __MACOSX

# The Kaggle archive nests an extra chest_xray/ folder inside itself --
# flatten it so the layout always matches what the code expects.
if [ -d "chest_xray/chest_xray" ]; then
  echo "Flattening the archive's extra nested chest_xray/ folder..."
  mv chest_xray/chest_xray/* chest_xray/
  rm -rf chest_xray/chest_xray
fi

echo "Done. Dataset is ready at $DEST_DIR/chest_xray"
