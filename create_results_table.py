import argparse
import os
import re
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--runs_dir", help="Run dir")
args = parser.parse_args()

assert args.runs_dir, "Run directory has to be provided"

runs_dir = args.runs_dir

print(f"\nThe provided directory containing the runs is: \n {runs_dir}\n")

city_names = runs_dir.split("/")[-2].split("_")[0].split("-")
city_names_test_auc = [name + "_test_auc" for name in city_names]
columns = ["run_id", "batch_size", "filter", "dropouts", "units", "learning_rate", "test_auc", "test_precision"] + city_names_test_auc


run_ids = []
batch_sizes = []
filters = []
dropouts =  []
units = []
learning_rates = []
test_aucs = []
test_precisions = []

city_test_aucs = {}  # Create an empty dictionary to store the lists

for city in city_names_test_auc:
    city_test_aucs[city] = []
    
for dir in os.listdir(runs_dir):
    if dir.isdigit():
        for file in os.listdir(runs_dir + "/" + dir):
            if file == "metadata.txt":
                fp = open(f"{runs_dir}/{dir}/metadata.txt")
                for i, line in enumerate(fp):
                
                    if i == 2:
                        match_run_id = re.search(r'Run (\d+)', line)
                        if match_run_id:
                            run_id = match_run_id.group(1)
                            assert dir == run_id, "There is a mismatch between the folder name and the run id in the metadata file!"
                            run_ids.append(run_id)
                        
                    if i == 10:

                        # Batch Size
                        match_batch_size = re.search(r'batch_size=([\d.e-]+)', line)
                        if match_batch_size:
                            batch_size = match_batch_size.group(1)
                            batch_sizes.append(batch_size)

                        # Filter
                        match_filter = re.search(r'filters=([\d.e-]+)', line)
                        if match_filter:
                            filter = match_filter.group(1)
                            filters.append(filter)

                        # Dropouts
                        match_dropout = re.search(r'dropout=([\d.e-]+)', line)
                        if match_dropout:
                            dropout = match_dropout.group(1)
                            dropouts.append(dropout)

                        # Units
                        match_units = re.search(r'units=([\d.e-]+)', line)
                        if match_units:
                            unit = match_units.group(1)
                            units.append(unit)
                        
                        # Learning Rate
                        match_learning_rate = re.search(r'learning_rate=([\d.e-]+)', line)
                        if match_learning_rate:
                            learning_rate = match_learning_rate.group(1)
                            learning_rates.append(learning_rate)
                        
                    if i == 14:
                        # Test AUC
                        match_auc = re.search(r'ROC Curve: ([\d.e-]+)', line)
                        if match_auc:
                            auc = match_auc.group(1)
                            test_aucs.append(auc)
                        
                    if i == 15:
                        # Test Precision
                        match_precision = re.search(r'precision:  ([\d.e-]+)', line)
                        if match_precision:
                            precision = match_precision.group(1)
                            test_precisions.append(precision)
                    
                    if i > 15:
                        match_city_auc = re.search(r'- ([\w]+) - test_auc: ([\d.e-]+)', line)
                        if match_city_auc:
                            city_auc_name = match_city_auc.group(1)
                            city_auc_name_column = city_auc_name + "_test_auc"
                            city_auc = match_city_auc.group(2)
                            city_test_aucs[city_auc_name_column].append(city_auc)                            
                    
                fp.close()

assert (len(run_ids) == len(batch_sizes)) & (len(run_ids) == len(filters)) & (len(run_ids) == len(dropouts)) & (len(run_ids) == len(units)) & (len(run_ids) == len(learning_rates)) & (len(run_ids) == len(test_aucs)) & (len(run_ids) == len(test_precisions)),  "All basic columns need to be of the same length"

for city_name in city_names_test_auc:
    assert len(run_ids) == len(city_test_aucs[city_name]), "Not all city level AUCs are of the correct length"
    
    
    
city_test_aucs[columns[0]] = run_ids
city_test_aucs[columns[1]] = batch_sizes
city_test_aucs[columns[2]] = filters
city_test_aucs[columns[3]] = dropouts
city_test_aucs[columns[4]] = units
city_test_aucs[columns[5]] = learning_rates
city_test_aucs[columns[6]] = test_aucs
city_test_aucs[columns[7]] = test_precisions


results = pd.DataFrame(city_test_aucs)
column_order_beginning = [results.columns[-8], results.columns[-7], results.columns[-6], results.columns[-5], results.columns[-4], results.columns[-3], results.columns[-1], results.columns[-2]] 
column_order_end = results.columns[:-9]

column_order = list(column_order_beginning) + list(column_order_end)
results = results[column_order]
results = results.sort_values(by="run_id")
results.to_csv(runs_dir + "results.csv", index=False)
print(f"The file 'results.csv' was successfully created!")
