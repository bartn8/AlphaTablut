#!/bin/bash

if [[ $# -eq 0 ]]; then
    player="W"
    timeout=59
    host="localhost"
elif [[ $# -eq 1 ]]; then
    player=$1
    timeout=59
    host="localhost"
elif [[ $# -eq 2 ]]; then
    player=$1
    timeout=$2
    host="localhost"
else
    player=$1
    timeout=$2
    host=$3
fi

cd "/tablut/AlphaTablut"
python3 client.py -p $player -t $timeout -i $host