import argparse
from pathlib import Path

import pandas as pd

from pea_classification import dataset_util

def parse_path(path_string: str) -> Path:
    path_string = Path(path_string).resolve()
    return path_string

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('training_data', type=parse_path, 
                        help='File path to the training data')
    parser.add_argument('test_data', type=parse_path, 
                        help='File path to the test data')
    parser.add_argument('output_file', type=parse_path, 
                        help='File path to store the generated data statistics')
    parser.add_argument('output_format', type=str, choices=['excel', 'csv'], 
                        help='Format of the output file')
    args = parser.parse_args()

    # Load data
    train = pd.read_excel(args.training_data)
    test = pd.read_excel(args.test_data)
    # Link dataset function with experiment ID number
    function_experiment_id = {dataset_util.performance_sentence_dataset: '9',
                              dataset_util.performance_sentence_wo_attribution_dataset: '10'}
    # Create the dataset subsets for each experiment
    dataset_stats_dfs = []
    for dataset_function, experiment_id in function_experiment_id.items():
        train_subset = dataset_function(train)
        test_subset = dataset_function(test)
        dataset_stats = dataset_util.data_stats(train_subset, test_subset, 
                                                experiment_id, 'Experiment ID')
        dataset_stats_dfs.append(dataset_stats)
    dataset_stats_df = pd.concat(dataset_stats_dfs, axis=1)
    # Save the data
    output_fp = args.output_file
    output_fp.parent.mkdir(parents=True, exist_ok=True)
    if args.output_format == 'excel':
        dataset_stats_df.to_excel(output_fp)
    else:
        dataset_stats_df.to_csv(output_fp)
        
