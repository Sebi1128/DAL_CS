#!/usr/bin/env bash

# This code runs main.py for all the yaml files in a folder successively.

# Usage:
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