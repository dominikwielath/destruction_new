#!/bin/bash
# source ../environments/destr/bin/activate
source env/bin/activate

# declare -a Cities=("aleppo" "daraa" "damascus" "deir-ez-zor" "hama" "homs" "idlib" "raqqa")
declare -a Cities=("aleppo" "daraa")
# declare -a Cities=("moschun" "volnovakha")
# declare -a Cities=('hostomel' 'irpin' 'kharkiv' 'livoberezhnyi' 'moschun' 'rubizhne' 'volnovakha')
# declare -a Cities=("aleppo")

declare -a data_dir=$1
echo "Data Dir: $data_dir";

for city in "${Cities[@]}"; do
    printf "\n"
    echo "#### Sampling:" $city
    printf "\n"
    python3 sample.py --city $city --data_dir $data_dir
    printf "\n"
    echo "#### Labeling:" $city
    printf "\n"
    python3 label.py --city $city --data_dir $data_dir
    printf "\n"
    echo "#### Tiling:" $city
    printf "\n"
    python3 tile.py --city $city --data_dir $data_dir
    printf "\n"
    echo "#### Balancing:" $city
    printf "\n"
    python3 balance.py --city $city --data_dir $data_dir
    printf "\n"
    echo "#### Shuffling:" $city
    printf "\n"
    python3 shuffle.py --city $city --data_dir $data_dir --block_size 10000
    printf "\n"
    echo "#### Shuffling again:" $city
    printf "\n"
    python3 shuffle.py --city $city --data_dir $data_dir --block_size 20000
done
