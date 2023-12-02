#!/bin/bash

# Define output folder and cities (desired runs subdirectory name)
# Artemisa
output_path="/lustre/ific.uv.es/ml/iae091/outputs" # !!! Stop before the "/runs" and "/data" subdirectories !!!
data_path="/lustre/ific.uv.es/ml/iae091/data/"
runs_subdirectory="hostomel-irpin-kharkiv-livoberezhnyi-moschun-rubizhne-volnovakha-aleppo-damascus-daraa-deirezzor-hama-homs-idlib-raqqa_8"

# Work Station 
# output_path="/media/andre/Samsung8TB/mwd-latest/outputs" # !!! Stop before the "/runs" and "/data" subdirectories !!!
# data_path="/media/andre/Samsung8TB/mwd-latest/data"
# runs_subdirectory="aleppo-hostomel-irpin-kharkiv-livoberezhnyi-moschun-rubizhne-volnovakha_1"




# Define an array to hold the city names
declare -a Cities
city_list=""

# Remove the suffix part
city_part="${runs_subdirectory%_*}"

# Split the remaining string by hyphen into an array
IFS='-' read -ra city_array <<< "$city_part"

# Add the city names to the Cities array
for city in "${city_array[@]}"; do
    Cities+=("$city")
    
    [ -n "$city_list" ] && city_list+=","
    city_list+="$city"
done

printf "Cities in this run: %s\n" "${Cities[*]}"

output_dir="$output_path/runs/$runs_subdirectory"


# Define a function to handle the interrupt signal
interrupt_handler() {
    echo "Script interrupted. Exiting..."
    exit 1
}

# Set up the trap to catch SIGINT (Ctrl+C) and call the interrupt_handler function
trap interrupt_handler SIGINT


##############
# Grid-Search#
##############

# Get the parameter table of the gridsearch
input_file="$output_dir/parameter_table.txt"

# Check if the parameter table exists
if [ ! -e "$input_file" ]; then
    echo "File does not exist: $input_file"
    echo "Please first specify the cities and parameters in create_parameter_table.sh and run it!"
    exit 1  # Exit with a non-zero status code
fi

echo "File exists: $input_file"


# Get the names of all run directories
directory_names=($(find $output_dir -type d -regex '.*/[0-9]+$' -exec basename {} \;))
printf "IDs for existing runs: %s\n" "${directory_names[*]}"


directory_names=($(find $output_dir -type d -regex '.*/[0-9]+$';))



# Identify the last run that was finished (contains file with name 'training.png'
highest_subdir="-1"
highest_value=-1

for subdir in "${directory_names[@]}"; do
    subdir_name=$(basename "$subdir")
    
    if [[ -f "$subdir/training.png" ]]; then
        current_value="${subdir_name//[!0-9]/}"
        if [[ $current_value -gt $highest_value ]]; then
            highest_value=$current_value
            highest_subdir=$subdir_name
        fi
    fi
done

# Deleting run higher than the last run but uncompleted
for subdir in "${directory_names[@]}"; do
    subdir_name=$(basename "$subdir")
    current_value="${subdir_name//[!0-9]/}"
    if [[ ! -f "$subdir/training.png" && $current_value -gt $highest_value ]];then
        echo "Deleting subdirectory: $subdir_name"
        rm -r "$subdir"
    fi
done



if [[ $highest_value != -1 ]]; then
	echo "Subdirectory of the completed run with the highest id: $highest_subdir"
else
	echo "No completed run in directory yet."
fi


# Walk through the parameters table starting with the highest id plus one
while IFS=" | " read -r id dropout lr filter unit; do
    if [[ "$id" != "ID" && "$id" != "-----------------------------------------------" && $id -gt $highest_value ]]; then
    	echo "Next run to start:"
        echo "ID: $id"
        echo "Dropout: $dropout"
        echo "Learning Rate: $lr"
        echo "Filters: $filter"
        echo "Units: $unit"
        echo "-------------------"
        echo ""
        
        python3 -u train_corrected.py --model double --cities $city_list --runs_dir $runs_subdirectory --run_id $id --dropout $dropout --lr $lr --filters $filter --units $unit --data_dir $data_path --output_dir $output_path 
    fi
done < "$input_file"


