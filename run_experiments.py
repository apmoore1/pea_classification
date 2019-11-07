import argparse
from pathlib import Path

import pandas as pd
import joblib
from sklearn.preprocessing import MaxAbsScaler

import pea_classification
from pea_classification.dataset_util import get_experiment_data
from pea_classification import classifiers
from pea_classification.classifier_util import classifier_grid_search, word_list_cv_scoring
from pea_classification.classifiers import WordListClassifier

def parse_path(path_string: str) -> Path:
    path_string = Path(path_string).resolve()
    return path_string

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("scores_dir", type=parse_path, 
                        help="Top level directory to store the results")
    parser.add_argument("models_dir", type=parse_path,
                        help="Top level directory to store the best models")
    parser.add_argument('accuracy_results_fp', type=parse_path, 
                        help='File path to store the aggregate results for the best model based on accuracy')
    parser.add_argument('accuracy_raw_results_fp', type=parse_path, 
                        help='File path to store the raw results for the best model based on accuracy')
    parser.add_argument('macro_f1_results_fp', type=parse_path, 
                        help='File path to store the aggregate results for the best model based on macro f1')
    parser.add_argument('macro_f1_raw_results_fp', type=parse_path, 
                        help='File path to store the raw results for the best model based on macro f1')
    args = parser.parse_args()

    training_data = pea_classification.dataset_util.train_dataset()
    test_data = pea_classification.dataset_util.test_dataset()

    # Tokenise text
    tokeniser = pea_classification.dataset_util.spacy_tokenizer()
    training_data['tokeniser_sentences'] = training_data.loc[:, 'sentence'].apply(lambda x: ' '.join(tokeniser(x)))
    test_data['tokeniser_sentences'] = test_data.loc[:, 'sentence'].apply(lambda x: ' '.join(tokeniser(x)))

    # Experiments
    experiment_ids = ['exp1', 'exp2', 'exp3', 'exp4', 'exp5', 'exp6a',
                      'exp6b', 'exp7a', 'exp7b', 'exp8', 'exp9', 'exp10a', 
                      'exp10b', 'exp11a', 'exp11b']
    sampling_keys = ['over_sampled', 'under_sampled', 'un_balanced']
    sampling_values = {'over_sampled': {'oversample': True, 'undersample': False},
                       'under_sampled': {'oversample': False, 'undersample': True},
                       'un_balanced': {'oversample': False, 'undersample': False}}
    classifier_names = ['complement_naive_bayes', 'multinomial_naive_bayes',
						'random_forest', 'svm_classifier', 'mlp_classifier']
    for experiment_id in experiment_ids:
        print(f'Starting experiment {experiment_id}')
        experiment_training_data = get_experiment_data(training_data, experiment_id, 
                                                       'tokeniser_sentences')
        sentences, labels = experiment_training_data
        for sampling_key in sampling_keys:
            print(f'Starting sampling {sampling_key}')
            sampling_value = sampling_values[sampling_key]
            for classifier_name in classifier_names:
                print(f'Starting classifier {classifier_name}')
                score_save_fp = Path(args.scores_dir, f'{experiment_id}', 
                                     f'{sampling_key}', f'{classifier_name}', 
                                     'train.csv')
                # We are saving the best model with regards to accuracy
                model_save_fp = Path(args.models_dir, 'accuracy', f'{experiment_id}', 
                                     f'{sampling_key}', f'{classifier_name}.joblib')
                if score_save_fp.exists() and model_save_fp.exists():
                    continue
                oversample = sampling_value['oversample']
                undersample = sampling_value['undersample']
                classifier_grid_search(sentences, labels, classifier_name, 40, 
                                       oversample, undersample, score_save_fp, 
                                       model_save_fp, 10, -1)
    # Word List experiments
    # For the word lists we are not performing any extra sampling/balancing 
    # as they do not require training data but we perform the same experiment 
    # multiple times so that the analysis of the results is easier.
    
    sentiment_lists = ['L&M', 'HEN_08', 'HEN_06']
    attribute_type = ['MW_TYPE']
    attribution = ['ZA_2015', 'Dikoli_2016', 'MW_ALL', 'MW_50']
    word_list_names = sentiment_lists + attribute_type + attribution
    experiment_word_lists = {'exp1': sentiment_lists, 'exp2': sentiment_lists, 
                             'exp3': attribution, 'exp4': attribution, 
                             'exp5': attribute_type, 'exp6a': attribution, 
                             'exp6b': attribution, 'exp7a': attribution, 
                             'exp7b': attribution, 'exp10a': sentiment_lists, 
                             'exp10b': sentiment_lists, 'exp11a': sentiment_lists, 
                             'exp11b': sentiment_lists}
    # Scores to measure like in the classifier experiments
    
    for experiment_id, word_lists in experiment_word_lists.items():
        print(f'Starting experiment {experiment_id}')
        # Compared to the machine learning methods here no tokenisation is required.
        experiment_training_data = get_experiment_data(training_data, experiment_id, 
                                                       'sentence')
        sentences, labels = experiment_training_data
        for sampling_key in sampling_keys:
            print(f'Starting sampling {sampling_key}')
            for word_list in word_lists:
                print(f'Starting word list {word_list}')
                score_save_fp = Path(args.scores_dir, f'{experiment_id}', 
                                     f'{sampling_key}', f'{word_list}', 
                                     'train.csv')
                model_save_fp = Path(args.models_dir, 'accuracy', f'{experiment_id}', 
                                     f'{sampling_key}', f'{word_list}.joblib')
                macro_model_save_fp = Path(args.models_dir, 'macro_f1', f'{experiment_id}', 
                                               f'{sampling_key}', f'{word_list}.joblib')
                if not macro_model_save_fp.exists():
                    if experiment_id == 'exp10b' or experiment_id == 'exp11b':
                        word_list_clf = WordListClassifier(word_list, pos_label=2, neg_label=1)
                    else:
                        word_list_clf = WordListClassifier(word_list)
                    word_list_clf.fit([])
                    macro_model_save_fp.parent.mkdir(parents=True, exist_ok=True)
                    joblib.dump(word_list_clf, str(macro_model_save_fp.resolve()))

                if score_save_fp.exists() and model_save_fp.exists():
                    continue
                if experiment_id == 'exp10b' or experiment_id == 'exp11b':
                    word_list_cv_scoring(sentences, labels, word_list, score_save_fp, 
                                        model_save_fp, cv=10, n_jobs=None, 
                                        pos_label=2, neg_label=1)
                else:
                    word_list_cv_scoring(sentences, labels, word_list, score_save_fp, 
                                        model_save_fp, cv=10, n_jobs=None)

    

    metric_name_2_column_name = {'Mean Accuracy': 'mean_test_accuracy', 
                                'SD Accuracy': 'std_test_accuracy', 
                                'F1 Class 1': 'mean_test_f1 class 1', 
                                'F1 Class 2': 'mean_test_f1 class 2',
                                'Macro F1': 'mean_test_macro_f1'}
    re_name_methods = {'complement_naive_bayes': 'C_NB', 
                       'multinomial_naive_bayes': 'M_NB',
					   'random_forest': 'RF', 'svm_classifier': 'SVM', 
                       'mlp_classifier': 'MLP'}
    
    for overview_fp, raw_fp, sorting_metric_name in [(args.accuracy_results_fp, args.accuracy_raw_results_fp, 'mean_test_accuracy'), 
                                                     (args.macro_f1_results_fp, args.macro_f1_raw_results_fp, 'mean_test_macro_f1')]:
        all_scores = []
        all_metrics = []
        all_balanced = []
        all_method_name = []
        all_experiment_ids = []
        for experiment_id in args.scores_dir.iterdir():
            for sampling_key in experiment_id.iterdir():
                for method_name in sampling_key.iterdir():
                    train_data_fp = Path(method_name, 'train.csv')
                    if not train_data_fp.exists():
                        raise ValueError(f'Cannot find the train data file {train_data_fp}')
                    train_data = pd.read_csv(train_data_fp)
                    if not 'mean_test_macro_f1' in train_data.columns:
                        f1_score_1 = train_data.loc[:, 'mean_test_f1 class 1']
                        f1_score_2 = train_data.loc[:, 'mean_test_f1 class 2']
                        macro_f1 = (f1_score_1 + f1_score_2) / 2
                        train_data['mean_test_macro_f1'] = macro_f1
                        # Save the Macro F1 scores
                        train_data.to_csv(train_data_fp)
                    sorted_train_data = train_data.sort_values(f'{sorting_metric_name}').copy()
                    if sorting_metric_name == 'mean_test_macro_f1':
                        model_save_fp = Path(args.models_dir, 'macro_f1', f'{experiment_id.name}', 
                                             f'{sampling_key.name}', f'{method_name.name}.joblib')
                        if not model_save_fp.exists():
                            print(method_name.name)
                            if method_name.name in word_list_names:
                                pass
                            else:
                                classifier = getattr(classifiers, method_name.name)()
                                pipeline = classifiers.classifier_pipeline(oversample=False, undersample=False, 
											                               classifier=classifier)
                                if sampling_key.name == 'under_sampled':
                                    print('under')
                                    pipeline = classifiers.classifier_pipeline(oversample=False, undersample=True, 
											                               classifier=classifier)
                                elif sampling_key.name == 'over_sampled':
                                    print('over')
                                    pipeline = classifiers.classifier_pipeline(oversample=True, undersample=False, 
											                               classifier=classifier)
                                print(experiment_id.name)
                                print(sampling_key.name)
                                best_macro_params = eval(sorted_train_data.loc[:, 'params'].iloc[-1])
                                pipeline.set_params(**best_macro_params)
                                experiment_training_data = get_experiment_data(training_data, experiment_id.name, 
                                                       'tokeniser_sentences')
                                sentences, labels = experiment_training_data
                                pipeline.fit(sentences, labels)
                                model_save_fp.parent.mkdir(parents=True, exist_ok=True)
                                joblib.dump(pipeline, str(model_save_fp.resolve()))
                    for metric_name, column_name in metric_name_2_column_name.items():
                        score = sorted_train_data.loc[:, column_name].iloc[-1] * 100
                        score = round(score, 2)
                        all_scores.append(score)
                        all_metrics.append(metric_name)

                        classifier_name = method_name.name
                        if classifier_name in re_name_methods:
                            classifier_name = re_name_methods[classifier_name]
                        all_method_name.append(classifier_name)
                        all_balanced.append(sampling_key.name)
                        all_experiment_ids.append(experiment_id.name)
        results_dict = {'Metric': all_metrics, 'Scores': all_scores, 
                        'Sampling': all_balanced, 'Method': all_method_name, 
                        'Experiment ID': all_experiment_ids}
        result_df = pd.DataFrame(results_dict)
        result_df.to_csv(raw_fp)
        pivot_df = pd.pivot_table(result_df, values='Scores', 
                                  columns=['Sampling', 'Metric', 'Method'], 
                                  index='Experiment ID')
        pivot_index = ['exp1', 'exp2', 'exp3', 'exp4', 'exp5', 'exp6a', 'exp6b',
                       'exp7a', 'exp7b', 'exp8', 'exp9', 'exp10a', 'exp10b', 
                       'exp11a', 'exp11b']
        pivot_df = pivot_df.reindex(pivot_index)
        overview_fp.parent.mkdir(parents=True, exist_ok=True)
        pivot_df.to_excel(overview_fp)
                
            


