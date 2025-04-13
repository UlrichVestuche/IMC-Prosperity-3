#!/bin/bash

# File to be modified and tested
FILE="Round_2/Gleb/PB1_ga.py"
# Log file where output will be collected
LOGFILE="prosperity3bt_PB2_test600-1900.txt"

# Define an array of fair values to test.
z_max=(1.9 2.0 2.1 2.2 2.3 2.4)

# Clear the log file
> "$LOGFILE"

# Outer loop: iterate over len_long values from 100 to 1000 in steps of 100.
for len in $(seq 600 100 1900); do
    echo "========================================" | tee -a "$LOGFILE"
    echo "Updating file with len_long = $len" | tee -a "$LOGFILE"
    
    # Update len_long value in the target file.
    sed -i.bak -E "s/^(len_long2[[:space:]]*=[[:space:]]*).*$/len_long2 = $len/" "$FILE"
    
    # Inner loop: iterate over z_max values.
    for z in "${z_max[@]}"; do
        echo "----------------------------------------" | tee -a "$LOGFILE"
        echo "Updating file with z_max2 = $z" | tee -a "$LOGFILE"
        
        # Update z_max value in the target file.
        sed -i.bak -E "s/^(z_max2[[:space:]]*=[[:space:]]*).*$/z_max2 = $z/" "$FILE"
        
        echo "File updated. Running prosperity3bt with z_max2 = $z, len_long2 = $len" | tee -a "$LOGFILE"
        
        # Run the trading algorithm via prosperity3bt (adjust the command if needed).
        prosperity3bt "$FILE" 2-1 --no-progress --no-out >> "$LOGFILE" 2>&1
        
        echo "Completed run for z_max2 = $z, len_long = $len" | tee -a "$LOGFILE"
    done
done

# Display the collected logs
echo "======== Combined Logs ========"
cat "$LOGFILE"