#!/bin/bash

echo "Starting script execution sequence..."

./run_p3ls.sh  # P3LS
# ./run_ppsr.sh  # PPSR  - Notes: This experiment took a really long time
./run_secureboost.sh  # Secureboost
./run_splitnn.sh  # SplitNN

echo "All scripts executed."

echo "Press any key to exit..."
read -n 1 -s -r -p ""
echo "Exiting..."