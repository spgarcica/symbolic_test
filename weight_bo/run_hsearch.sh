#!/bin/bash

basenames=$(awk '{ print }' basenames.txt)

for i in ${basenames[@]}
do
	echo "===== testing dataset: $i ====="
	python ./hsearch.py --name $i --data_path $i --num_trials 1
done
	
