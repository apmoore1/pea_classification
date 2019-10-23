from collections import Counter
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd

def remove_blank_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    '''
    Removes rows of data that contain the `BLANK` and `DUPLICATE` values  
    in the `senttype` column.

    :param df: DataFrame containing the PEA dataset
    :returns: The PEA Dataset with the `BLANK` and `DUPLICATE` values 
            removed from the `senttype` column.
    '''
    unwanted_values = ['BLANK', 'DUPLICATE']
    return df[~df['senttype'].isin(unwanted_values)]


def performance_sentence_dataset(df: pd.DataFrame
                                 ) -> Tuple[List[str], np.ndarray, 
                                            Dict[int, str]]:
    '''
    Dataset of performance sentences containing sentence that do and do not 
    have attribution. The dataset relating to experiment EXP1.

    :param df: The training dataset as a dataframe
    :returns: A tuple containing 1. A list of strings representing sentences, 2.
              An array of 1's or 0's representing whether or not the associated 
              string is a performance sentence or not. 3. A mapper that maps 
              labels integers to the label's name.
    :raises ValueError: If the number of sentences does not equal the number 
                        of labels.
    '''
    subset_df = pd.DataFrame.copy(df)
    subset_df = remove_blank_duplicates(subset_df)
    performance_values = ['perfnonatt', 'attperf']
    non_performance_values = ['attnonperf', 'nonperfnonatt']
    performance_rows = subset_df[subset_df['senttype'].isin(performance_values)]
    performance_rows['y'] = 1
    non_performance_rows = subset_df[subset_df['senttype'].isin(non_performance_values)]
    non_performance_rows['y'] = 0
    all_rows = pd.concat([performance_rows, non_performance_rows])
    sentences = all_rows['Sentence'].tolist()
    labels = all_rows['y'].to_numpy()
    shape_err = (f'The number of sentences should {len(sentences)}, should equal '
                f'the number of labels {labels.shape[0]}')
    assert len(sentences) == labels.shape[0], shape_err
    label_mapper = {1: 'Performance', 0: 'Non-Performance'}
    return (sentences, labels, label_mapper)

def performance_sentence_wo_attribution_dataset(df: pd.DataFrame
                                                ) -> Tuple[List[str], np.ndarray, 
                                                           Dict[int, str]]:
    '''
    The dataset of performance sentences where the sentence does not have any 
    attribution. The dataset relating to experiment EXP2.

    :param df: The training dataset as a dataframe
    :returns: A tuple containing 1. A list of strings representing sentences, 2.
              An array of 1's or 0's representing whether or not the associated 
              string is a performance sentence or not. 3. A mapper that maps 
              labels integers to the label's name.
    :raises ValueError: If the number of sentences does not equal the number 
                        of labels.
    '''
    subset_df = pd.DataFrame.copy(df)
    subset_df = remove_blank_duplicates(subset_df)
    performance_values = ['perfnonatt']
    non_performance_values = ['nonperfnonatt']
    performance_rows = subset_df[subset_df['senttype'].isin(performance_values)]
    performance_rows['y'] = 1
    non_performance_rows = subset_df[subset_df['senttype'].isin(non_performance_values)]
    non_performance_rows['y'] = 0
    all_rows = pd.concat([performance_rows, non_performance_rows])
    sentences = all_rows['Sentence'].tolist()
    labels = all_rows['y'].to_numpy()
    shape_err = (f'The number of sentences should {len(sentences)}, should equal '
                 f'the number of labels {labels.shape[0]}')
    assert len(sentences) == labels.shape[0], shape_err
    label_mapper = {1: 'Performance W/O Attribution', 
                    0: 'Non-Performance W/O Attribution'}
    return (sentences, labels, label_mapper)

def data_stats(train_data: Tuple[List[str], np.ndarray, Dict[int, str]],
               test_data: Tuple[List[str], np.ndarray, Dict[int, str]],
               experiment_name: Optional[str] = None,
               experiment_name_label: str = 'Experiment ID'
               ) -> pd.DataFrame:
    '''
    :param train_data: The training data, return of any of the dataset functions
    :param test_data: The test data, return of any of the dataset functions
    :param experiment_name: The name of the experiment or the experiment ID. If
                            not None then this will be the hierarchical column 
                            on top of the class names.
    :param experiment_name_label: The name to give to the experiment_name 
                                  column that will be shown in the pivot 
                                  table. 
    :returns: A pivot table containing the dataset name as index, columns 
              represent the class label and the values are the number of 
              class labels in each dataset.
    '''
    def label_name_counts(data: Tuple[List[str], np.ndarray, Dict[int, str]]
                          ) -> Dict[str, int]:
        '''
        :param data: The return of any of the dataset functions
        :returns: A dictionary mapping the label/class name to the number of times 
                  that label/class has appeared in the dataset.
        '''
        _, labels_array, label_mapper = data
        label_count = Counter(labels_array)
        label_name_count = {label_mapper[label_int]: count 
                            for label_int, count in label_count.items()}
        return label_name_count

    label_names = []
    label_counts = []
    label_norm_counts = []
    dataset_names = []
    for data, name in [(train_data, 'Train'), (test_data, 'Test')]:
        total_count = 0
        data_label_counts = []
        for label_name, count in label_name_counts(data).items():
            total_count += count
            label_names.append(label_name)
            data_label_counts.append(count)
            dataset_names.append(name)
        label_counts.extend(data_label_counts)
        for label_count in data_label_counts:
            label_norm_count = (label_count / total_count) * 100
            label_norm_counts.append(label_norm_count)
    df_dict = {'Label Count': label_counts, 
               'Label Norm Counts': label_norm_counts, 
               'Label': label_names, 'Dataset': dataset_names}
    if experiment_name is not None:
        df_dict[experiment_name_label] = [experiment_name] * len(label_names)
    df = pd.DataFrame(df_dict)
    label_count_series = df[['Label Count', 'Label Norm Counts']]
    count_percent_func = lambda x: f'{x[0]} ({x[1]:.2f}%)'
    df['Label Count (%)'] = label_count_series.apply(count_percent_func, axis=1)
    if experiment_name is not None:
        return pd.pivot_table(df, index='Dataset', values='Label Count (%)', 
                              columns=[experiment_name_label, 'Label'], 
                              aggfunc=lambda x: ' '.join(x))
    else:
        return pd.pivot_table(df, index='Dataset', values='Label Count (%)', 
                              columns='Label', aggfunc=lambda x: ' '.join(x))