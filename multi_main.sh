#!/usr/bin/env bash

# Deep Active Learning with Contrastive Sampling
#
# Deep Learning Project for Deep Learning Course (263-3210-00L)  
# by Department of Computer Science, ETH Zurich, Autumn Semester 2021 
#
# Authors:  
# Sebastian Frey (sefrey@student.ethz.ch)  
# Remo Kellenberger (remok@student.ethz.ch)  
# Aron Schmied (aronsch@student.ethz.ch)  
# Guney Tombak (gtombak@student.ethz.ch)  

# This code runs main.py for all the yaml files in a folder successively.

# Usage:
# conda activate dalcs
# chmod +x multi_main.sh
# ./multi_main.sh -c <folder_path>

# Parse options and extract values to variables
while getopts "c:" opt; do
	case ${opt} in
		c )
		 	target=${OPTARG}
			;;
		\? )
			echo "Usage: cmd [-c] "
			;;
		: )
      		echo "Invalid option: $OPTARG requires an argument" 1>&2
      		return
      		;;
	esac
done
shift $((OPTIND -1)) # Remove options that have already been handled 

for case in ${target}/*.yaml; do
	echo "STARTING: ${case}"
	python3 main.py --config ${case}
	echo "FINISHING: ${case}"
done