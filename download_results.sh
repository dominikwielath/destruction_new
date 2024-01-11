#!/bin/bash

remote_server="iae0911@mlui01.ific.uv.es"
artemisa_runs_dir="/lustre/ific.uv.es/ml/iae091/outputs/runs/irpin-kharkiv-livoberezhnyi-moschun-rubizhne-volnovakha-aleppo-damascus-daraa-deirezzor-hama-homs-idlib-raqqa_1"
local_runs_dir="/home/dominik/Documents/ra_hannes/war-destruction/artemisa-results/irpin-kharkiv-livoberezhnyi-moschun-rubizhne-volnovakha-aleppo-damascus-daraa-deirezzor-hama-homs-idlib-raqqa_1"

echo "The following files/directories exist locally:"
echo "$(ls $local_runs_dir)"

# SSH into the remote server and get a list of subdirectories
subdirectories=$(ssh "$remote_server" "cd $artemisa_runs_dir && find . -maxdepth 2 -mindepth 2 -type f -name 'training.png' -exec dirname {} \;")

echo "The following directories of completed runs exist remotely:"
echo "$subdirectories"

for subdirectory in $subdirectories; do
    # Remove the leading './' from the subdirectory path
    subdirectory=${subdirectory#./}

    #Check if the subdirectory exists locally
    if [ -d "$local_runs_dir/$subdirectory" ]; then
        echo "Local subdirectory $subdirectory exists. Skipping..."
    else
        echo "Downloading $subdirectory..."
        mkdir -p "$local_runs_dir/$(dirname $subdirectory)"
        scp -r "$remote_server:$artemisa_runs_dir/$subdirectory" "$local_runs_dir/$subdirectory"
        echo "Downloaded $subdirectory"
        
    fi
done

echo "All subdirectories checked and downloaded if needed.\n"

#scp "$remote_server:$artemisa_runs_dir/parameter_table.txt" "$local_runs_dir/parameter_table.txt"

scp "$remote_server:$artemisa_runs_dir/results.csv" "$local_runs_dir/results.csv"
echo "Successfully downloaded the results table"
