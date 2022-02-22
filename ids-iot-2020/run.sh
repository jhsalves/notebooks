#!/bin/bash

for i in `seq 1 20`; do
    echo "Starting client $i"
    python3 MembershipInference.py  --sample=${i} &
done

# This will allow you to use CTRL+C to stop all background processes
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM
# Wait for all background processes to complete
wait
