import argparse
import os
import re
import pandas as pd
parser = argparse.ArgumentParser()
parser.add_argument("run_id", help="Model Run ID for which we want to generate predictions")
parser.add_argument("--cities", help="Pre File")
parser.add_argument("--data_dir", help="Model Run ID for which we want to generate predictions")
parser.add_argument("--output_dir", help="Model Run ID for which we want to generate predictions")
args = parser.parse_args()

"""
USAGE:
python -m predict 3 aleppo,daraa
"""


## For local
# CITIES = ['aleppo', 'daraa']
# OUTPUT_DIR = "../data/destr_outputs"
# DATA_DIR = "../data/destr_data"

# For artemisa
# CITIES = ['hostomel', 'irpin', 'kharkiv', 'livoberezhnyi', 'moschun', 'rubizhne', 'volnovakh, 'aleppo', 'damascus', 'daraa', 'deirezzor','hama', 'homs', 'idlib', 'raqqa']
CITIES = ['moschun', 'irpin']
OUTPUT_DIR = "/lustre/ific.uv.es/ml/iae091/outputs/runs/hostomel-irpin-kharkiv-livoberezhnyi-moschun-rubizhne-volnovakha-aleppo-damascus-daraa-deirezzor-hama-homs-idlib-raqqa_1"
DATA_DIR = "/lustre/ific.uv.es/ml/iae091/data"

# ## For workstation
# CITIES = ['aleppo', 'damascus', 'daraa', 'deir-ez-zor','hama', 'homs', 'idlib', 'raqqa']
# OUTPUT_DIR = "../outputs"
# DATA_DIR = "../data"


if args.data_dir:
    OUTPUT_DIR = args.data_dir

if args.output_dir:
    DATA_DIR = args.output_dir

if args.cities:
    CITIES = [el.strip() for el in args.cities.split(",")]


RUN_DIR = f'{OUTPUT_DIR}/{args.run_id}'




def search_data(pattern:str='.*', directory:str='../data') -> list:
    '''Sorted list of files in a directory matching a regular expression'''
    files = list()
    for root, _, file_names in os.walk(directory):
        for file_name in file_names:
            files.append(os.path.join(root, file_name))
    files = list(filter(re.compile(pattern).search, files))
    files.sort()
    if len(files) == 1: files = files
    return files

for city in CITIES:

    if os.path.exists(f"{RUN_DIR}/predictions_{city}.csv"):
        print("File already exists!")
        os.remove(f"{RUN_DIR}/predictions_{city}.csv")

    pre_images  = search_data(pattern='^.*tif', directory=f'{DATA_DIR}/{city}/images/pre')
    post_images  = search_data(pattern='^.*tif', directory=f'{DATA_DIR}/{city}/images/post')


    for pre_ in pre_images:
        for post_ in post_images:
            print(city, "-", pre_.split("/")[-1], post_.split("/")[-1])

            os.system(f"python -m predict_chunk {args.run_id} {pre_} {post_} --data_dir {DATA_DIR} --output_dir {OUTPUT_DIR}")

# Join files for all cities
file_list = os.listdir(OUTPUT_DIR)
pattern = r'predictions_[a-zA-Z]+\.csv'            

counter = 0
for file_name in file_list:
    if re.match(pattern, file_name):
        if counter == 0:
            predictions_all_cities = pd.read_csv(OUTPUT_DIR + "/" + file_name)
        else:
            predictions_this_city = pd.read_csv(OUTPUT_DIR + "/" + file_name)
            if (predictions_this_city.columns == predictions_all_cities.columns).all():
                predictions_all_cities = pd.concat([predictions_all_cities, predictions_this_city])
            else:
                print(f"The columns in file {file_name} did not match with the columns in the other files!")
        counter += 1
        
predictions_all_cities = predictions_all_cities.loc[predictions_all_cities["tr_va_te"] != "tr_va_te"]
predictions_all_cities = predictions_all_cities.reset_index(drop=True)
predictions_all_cities["tr_va_te"] = predictions_all_cities["tr_va_te"].astype(int)
predictions_all_cities["y"] = predictions_all_cities["y"].astype(float)
predictions_all_cities["yhat"] = predictions_all_cities["yhat"].astype(float)

predictions_csv = f"{OUTPUT_DIR}/predictions_all_cities.csv"

predictions_all_cities.to_csv(predictions_csv)
