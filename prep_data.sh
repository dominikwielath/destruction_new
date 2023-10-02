#!/bin/bash
# source ../environments/destr/bin/activate
source env/bin/activate

# declare -a Cities=("aleppo" "daraa" "damascus" "deir-ez-zor" "hama" "homs" "idlib" "raqqa")
# declare -a Cities=("aleppo" "daraa")
declare -a Cities=("moschun" "volnovakha")
#declare -a Cities=('hostomel' 'irpin' 'kharkiv' 'livoberezhnyi' 'moschun' 'rubizhne' 'volnovakha')
# declare -a Cities=("raqqa")

declare -a data_dir=$1
echo "Data Dir: $data_dir";

for city in "${Cities[@]}"; do
    echo "Sampling:" $city
    python3 sample.py --city $city --data_dir $data_dir
    echo "Labeling:" $city
#    python3 label.py --city $city --data_dir $data_dir
    python3 label_corrected.py --city $city --data_dir $data_dir
    echo "Tiling:" $city
    python3 tile.py --city $city --data_dir $data_dir
    echo "Balancing:" $city
    python3 balance.py --city $city --data_dir $data_dir
    echo "Shuffling:" $city
    python3 shuffle.py --city $city --data_dir $data_dir --block_size 10000
    echo "Shuffling again:" $city
    python3 shuffle.py --city $city --data_dir $data_dir --block_size 20000
done
