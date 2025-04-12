#!/bin/bash

# File to be modified and tested
FILE="Round_2/Gleb/PB1_ga.py"
# Log file where output will be collected
LOGFILE="prosperity3bt_logs.txt"

# Define an array of fair values to test.
z_max=(1 2)

# Clear the log file
> "$LOGFILE"

for z in "${z_max[@]}"; do
    echo "----------------------------------------" | tee -a "$LOGFILE"
    echo "Updating file with fair_value =$z" | tee -a "$LOGFILE"
    
    # Define the OLD_LINE (exact match for the original setting).
    OLD_LINE='z_max = 2'
    
    # Build the new line using the current fair value.
    NEW_LINE="z_max = $z"
    
    # Use sed to replace the line in the file.
    sed -i.bak "/^$(printf '%s\n' "$OLD_LINE" | sed 's:[\/&]:\\&:g')\$/c\\
$NEW_LINE" "$FILE"
    
    echo "File updated. Running prosperity3bt with fair_value = $z" | tee -a "$LOGFILE"
    
    # Run the trading algorithm via prosperity3bt (adjust the command if needed).
    prosperity3bt "$FILE" 2 >> "$LOGFILE" 2>&1
    
    echo "Completed run for fair_value = $z" | tee -a "$LOGFILE"
done

# Display the collected logs
echo "======== Combined Logs ========"
cat "$LOGFILE"
