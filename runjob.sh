#!/bin/bash
source /opt/conda/etc/profile.d/conda.sh
conda activate alphatablut
python main.py -a -s 0</dev/null 1>stdout.out 2>stderr.out &
