#!/bin/bash

# File to be modified and tested
FILE="Round_2/Gleb/PB1_ga.py"
# Log file where output will be collected
LOGFILE="prosperity3bt_logs.txt"

# Define an array of fair values to test.
z_max=(1.9 2 2.1 2.2 2.3 2.4 2.5 2.6 2.7 2.8 2.9 3)

# Clear the log file
> "$LOGFILE"

for z in "${z_max[@]}"; do
    echo "----------------------------------------" | tee -a "$LOGFILE"
    echo "Updating file with fair_value =$z" | tee -a "$LOGFILE"
    
    # Define the OLD_LINE (exact match for the original setting).
#     OLD_LINE='z_max = 1'
    
#     # Build the new line using the current fair value.
#     NEW_LINE="z_max = $z\n"
    
#     # Use sed to replace the line in the file.
#     sed -i.bak "/^$(printf '%s\n' "$OLD_LINE" | sed 's:[\/&]:\\&:g')\$/c\\
# $NEW_LINE" "$FILE"
    sed -i.bak -E "s/^(z_max[[:space:]]*=[[:space:]]*).*$/z_max = $z/" "$FILE"
    
    echo "File updated. Running prosperity3bt with z_max = $z" | tee -a "$LOGFILE"
    
    # Run the trading algorithm via prosperity3bt (adjust the command if needed).
    prosperity3bt "$FILE" 2 --no-progress   >> "$LOGFILE" 2>&1
    
    echo "Completed run for z_max = $z" | tee -a "$LOGFILE"
done

# Display the collected logs
echo "======== Combined Logs ========"
cat "$LOGFILE"
