import os
import re
import pandas as pd

PREDICTIONS_DIR = "/home/dominik/Documents/ra_hannes/war-destruction/artemisa-results/hostomel-irpin-kharkiv-livoberezhnyi-moschun-rubizhne-volnovakha-aleppo-damascus-daraa-deirezzor-hama-homs-idlib-raqqa_1/9_predictions/9/"

# Join files for all cities
file_list = os.listdir(PREDICTIONS_DIR)
pattern = r'predictions_[a-zA-Z]+\.csv'            

counter = 0
for file_name in file_list:
    if re.match(pattern, file_name):
        if counter == 0:
            predictions_all_cities = pd.read_csv(PREDICTIONS_DIR + file_name)
        else:
            predictions_this_city = pd.read_csv(PREDICTIONS_DIR + file_name)
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

predictions_csv = f"{PREDICTIONS_DIR}/predictions_all_cities.csv"

predictions_all_cities.to_csv(predictions_csv, index=False)
