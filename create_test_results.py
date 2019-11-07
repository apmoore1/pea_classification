import argparse
from collections import defaultdict
from pathlib import Path
from typing import Dict, Any, Optional

import pandas as pd

import pea_classification
from pea_classification.classifier_util import evaluate_methods, get_pre_trained_models, predict_score

def get_ids_sentence_labels(dataset: pd.DataFrame, ids: Dict[str, int], 
                            sentence_column: Optional[str] = None
                            ) -> Dict[str, Any]:
    '''
    :param dataset: The whole training or test dataset that must contain the 
                    following columns: 'DocumentID', 'SentenceID', 'SentmainID', 
                    'sentence', 'senttype', 'perftone', 'atttype'
    :param ids: A dictionary containing the following keys 
                'DocumentID', 'SentenceID', 'SentmainID' whose values must be 
                a list of integers where at each index for each key when combined 
                create a unique index
    :param sentence_column: Name of the sentence column. Default `sentence`
    :returns: Another dictionary that only contain the following keys 'DocumentID', 
                'SentenceID', 'SentmainID', 'sentence', 'senttype', 'perftone', 
                'atttype' and the values for these keys are a subset of the original
                dataset based on the values in the `ids` argument. In affect this 
                method subsets the `dataset` by the given `ids`.
    '''
    _ids_df = pd.DataFrame(_ids)
    keys = list(_ids_df.columns.values)

    alt_dataset = dataset.copy()
    alt_dataset_index = alt_dataset.set_index(keys).index
    _ids_index_df = _ids_df.copy().set_index(keys).index

    subset_alt_dataset = alt_dataset[alt_dataset_index.isin(_ids_index_df)].copy()
    if sentence_column is None:
        sentence_column= 'sentence'

    interested_columns = ['DocumentID', 'SentenceID', 'SentmainID', sentence_column, 
                           'senttype', 'perftone', 'atttype']
    subset_alt_dataset_rel = subset_alt_dataset.loc[:, interested_columns]
    subset_alt_dataset_rel_dict = subset_alt_dataset_rel.to_dict()
    rel_index = subset_alt_dataset_rel.index
    id_sent_labels_dict = defaultdict(list)
    for col in interested_columns:
        col_dict = subset_alt_dataset_rel_dict[col]
        if col == sentence_column:
            col = 'sentence'
        for ind in rel_index:
            id_sent_labels_dict[col].append(col_dict[ind])
    return id_sent_labels_dict

def parse_path(path_string: str) -> Path:
    path_string = Path(path_string).resolve()
    return path_string

if __name__ == '__main__':
    sentence_level_help = ("Whether the results should be outputted at the "
                          "sentence level with the classifiers stating if "
                          "they got the sentence correct or not.")
    metric_tuned_help = ("Given the metric it will return the best tuned "
                         "model for that metric based on the 10 fold cross "
                         "validation results")
    parser = argparse.ArgumentParser()
    parser.add_argument("result_fp", type=parse_path, 
                        help="File path to save test results too.")
    parser.add_argument("metric_tuned", type=str, 
                        help=metric_tuned_help)
    parser.add_argument("--sentence_level_results", action="store_true", 
                        help=sentence_level_help)
    args = parser.parse_args()
    args.result_fp.parent.mkdir(parents=True, exist_ok=True)
    if args.result_fp.exists():
        print(f'Results already exist {args.result_fp}')

    experiment_ids = ['exp1', 'exp2', 'exp3', 'exp4', 'exp5', 'exp6a',
                      'exp6b', 'exp7a', 'exp7b', 'exp8', 'exp9', 'exp10a', 
                      'exp10b', 'exp11a', 'exp11b']
    sampling_keys = ['over_sampled', 'under_sampled', 'un_balanced']
    classifier_names = ['complement_naive_bayes', 'random_forest', 
                        'svm_classifier', 'mlp_classifier']
    sentiment_lists = ['L&M', 'HEN_08', 'HEN_06']
    attribute_type = ['MW_TYPE']
    attribution = ['ZA_2015', 'Dikoli_2016', 'MW_ALL', 'MW_50']
    experiment_word_lists = {'exp1': sentiment_lists, 'exp2': sentiment_lists, 
                             'exp3': attribution, 'exp4': attribution, 
                             'exp5': attribute_type, 'exp6a': attribution, 
                             'exp6b': attribution, 'exp7a': attribution, 
                             'exp7b': attribution, 'exp10a': sentiment_lists, 
                             'exp10b': sentiment_lists, 'exp11a': sentiment_lists, 
                             'exp11b': sentiment_lists}
    method_2_expname = {'complement_naive_bayes': 'NB', 'random_forest': 'RF', 
                        'svm_classifier': 'SVM', 'mlp_classifier': 'MLP', 
                        'L&M': 'L&M', 'HEN_08': 'HEN_08', 'HEN_06': 'HEN_06', 
                        'MW_TYPE': 'MW_TYPE', 'ZA_2015': 'ZA_2015', 
                        'Dikoli_2016': 'Dikoli_2016', 'MW_ALL': 'MW_ALL', 
                        'MW_50': 'MW_50'}

    # Tokenise text
    test_data = pea_classification.dataset_util.test_dataset()
    tokeniser = pea_classification.dataset_util.spacy_tokenizer()
    test_data['tokeniser_sentences'] = test_data.loc[:, 'sentence'].apply(lambda x: ' '.join(tokeniser(x)))

    # Reason why we cannot run the machine learning classifiers at the same time 
    # as the word list approaches is that the classifier methods require tokenisation 
    # where as the word based do not.

    # Only required if sentence_level_results is True
    sheet_names_dataframe = {}
    # Only required if sentence_level_results is False
    all_scores = []
    all_metrics = []
    all_balanced = []
    all_method_name = []
    all_experiment_ids = []

    for experiment_id in experiment_ids:
        for sampling_key in sampling_keys:
            # This is to get around the duplicated experiments
            get_classifier_exp_id = experiment_id
            if experiment_id == 'exp7a':
                get_classifier_exp_id = 'exp6a'
            if experiment_id == 'exp7b':
                get_classifier_exp_id = 'exp6b'
            if experiment_id == 'exp11a':
                get_classifier_exp_id = 'exp10a'
            if experiment_id == 'exp11b':
                get_classifier_exp_id = 'exp10b'
            # Download relevant classifier
            classifier_methods = [get_pre_trained_models(classifier_name, sampling_key, get_classifier_exp_id, args.metric_tuned) 
                                  for classifier_name in classifier_names]
            relevant_classifier_names = [method_2_expname[classifier_name] for classifier_name in classifier_names]

            if args.sentence_level_results:
                sentences, labels, _ids = pea_classification.dataset_util.get_experiment_data(test_data, experiment_id, 'tokeniser_sentences', True)
                input_data = get_ids_sentence_labels(test_data, _ids, sentence_column='tokeniser_sentences')
                # Run the classifier methods
                classifier_results = evaluate_methods(input_data, labels, 
                                                      classifier_methods, relevant_classifier_names)
                if experiment_id in experiment_word_lists:
                    relevant_word_lists = experiment_word_lists[experiment_id]
                    # Download relevant word lists
                    word_methods = [get_pre_trained_models(word_list_name, sampling_key, get_classifier_exp_id, args.metric_tuned) 
                                    for word_list_name in relevant_word_lists]
                    relevant_word_names = [method_2_expname[word_list_name] for word_list_name in relevant_word_lists]
                    sentences, labels, _ids = pea_classification.dataset_util.get_experiment_data(test_data, experiment_id, 'sentence', True)
                    # Run the word list methods
                    word_results = evaluate_methods({'sentence': sentences, **_ids}, 
                                                    labels, word_methods, relevant_word_names)
                    classifier_results = pd.concat([classifier_results, word_results], axis=1)
                sheet_name = f'{experiment_id} {sampling_key}'
                sheet_names_dataframe[sheet_name] = classifier_results
            else:
                sentences, labels = pea_classification.dataset_util.get_experiment_data(test_data, experiment_id, 'tokeniser_sentences', False)
                for classifier_method, classifier_name in zip(classifier_methods, relevant_classifier_names):
                    classifier_scores = predict_score(classifier_method, sentences, labels)
                    for metric_name, score in classifier_scores.items():
                        all_scores.append(score)
                        all_metrics.append(metric_name)
                        all_balanced.append(sampling_key)
                        all_method_name.append(classifier_name)
                        all_experiment_ids.append(experiment_id)
                if experiment_id in experiment_word_lists:
                    relevant_word_lists = experiment_word_lists[experiment_id]
                    # Download relevant word lists
                    word_methods = [get_pre_trained_models(word_list_name, sampling_key, get_classifier_exp_id, args.metric_tuned) 
                                    for word_list_name in relevant_word_lists]
                    relevant_word_names = [method_2_expname[word_list_name] for word_list_name in relevant_word_lists]
                    sentences, labels = pea_classification.dataset_util.get_experiment_data(test_data, experiment_id, 'sentence', False)
                    for word_method, word_name in zip(word_methods, relevant_word_names):
                        word_scores = predict_score(word_method, sentences, labels)
                        for metric_name, score in word_scores.items():
                            all_scores.append(score)
                            all_metrics.append(metric_name)
                            all_balanced.append(sampling_key)
                            all_method_name.append(word_name)
                            all_experiment_ids.append(experiment_id)

    if args.sentence_level_results:
        with pd.ExcelWriter(args.result_fp) as writer:
            for sheet_name, dataframe in sheet_names_dataframe.items():
                dataframe.to_excel(writer, sheet_name=f'{sheet_name}')
    else:
        results_dict = {'Metric': all_metrics, 'Scores': all_scores, 
                        'Sampling': all_balanced, 'Method': all_method_name, 
                        'Experiment ID': all_experiment_ids}
        result_df = pd.DataFrame(results_dict)
        result_df.to_csv(args.result_fp)

