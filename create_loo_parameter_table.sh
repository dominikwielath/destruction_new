#!/bin/bash

# Define output folder, cities and parameters 
# Artemisa
output_path="/lustre/ific.uv.es/ml/iae091/outputs/runs"
declare -a Cities=('hostomel' 'irpin' 'kharkiv' 'livoberezhnyi' 'moschun' 'rubizhne' 'volnovakha' 'aleppo' 'damascus' 'daraa' 'deirezzor' 'hama' 'homs' 'idlib' 'raqqa')

# Work Station
# output_path="/media/andre/Samsung8TB/mwd-latest/outputs/runs"
# declare -a Cities=('aleppo' 'hostomel' 'irpin' 'kharkiv' 'livoberezhnyi' 'moschun' 'rubizhne' 'volnovakha')

dropout="0.10"
unit="64"
filter="64"
lr="0.00003" # Learning Rate



# Get the runs directory name and create the directory if it does not exist yet
# Join array elements with a dash (-)
joined_cities=$(IFS=-; echo "${Cities[*]}")

# Print the joined string
echo "Cities: $joined_cities"

# Get a list of existing subfolders with the same name
existing_folders=$(find "$output_path" -type d -name "${joined_cities}_*" | grep -E "${joined_cities}_[0-9]+$" | sort -r)

if [ -z "$existing_folders" ]; then
    # No existing folders, so create a new folder with suffix _1
    new_folder="${joined_cities}_1"
else
    # Get the highest suffix from existing folders
    highest_suffix=$(echo "$existing_folders" | head -n 1 | awk -F '_' '{print $NF}')
    
    # Increment the highest suffix and create a new folder
    new_suffix=$((highest_suffix + 1))
    new_folder="${joined_cities}_$new_suffix"
fi

echo "$highest_suffix"
mkdir "$output_path/$new_folder"
echo "Created subfolder: $new_folder"


output_file="$output_path/$new_folder/parameter_table.txt"

echo "ID | City Left Out | Dropout | Learning Rate | Filters | Units" > "$output_file"
echo "-----------------------------------------------" >> "$output_file"

id=0

for city in "${Cities[@]}"; do
	echo "$id | $city | $dropout | $lr | $filter | $unit" >> "$output_file"
	((id++))
done


echo "Table saved to $output_file"

