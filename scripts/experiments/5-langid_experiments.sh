#bin/bash

# Run the md3 experiments
cd ./code

# This code runs on cpu!  About 20-30 minutes for tulu-sft, 30-40 minutes for ultrachat

# Tulu SFT
python langid-tulu-sft.py --output_file=./outputs/tulu-sft-langid/langid.csv

# Ultrachat
python langid-ultrachat.py --output_file=./outputs/ultrachat-langid/langid.csv

cd ../