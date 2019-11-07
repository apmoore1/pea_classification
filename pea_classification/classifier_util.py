from typing import Optional, Tuple, List, Dict, Any, Union
from pathlib import Path

import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import RandomizedSearchCV, cross_validate
from sklearn.metrics import f1_score, accuracy_score, make_scorer
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline

from pea_classification import classifiers
from pea_classification.classifiers import WordListClassifier
from pea_classification.dataset_util import _get_from_cache

def predict_score(model: BaseEstimator, sentences: List[str], 
                  labels: np.ndarray) -> Dict[str, float]:
	'''
	:param model: A pre-trained classifier which can be either a word list 
	              classifier or bag of words machine learning classifier
	:param sentences: A list of sentences to be classified
	:param labels: The true labels associated to the given sentences
	:returns: A dictionary contaning: 1. Accuracy result with key `Accuracy`, 
	          2. F1 score for the 1 label with key `F1 Class 1`, 3. F1 
			  score for the 2 label with key `F1 Class 2`, and 4. Macro F1
			  score.
	'''
	results = {}
	
	predictions = model.predict(sentences)
	results['Accuracy'] = accuracy_score(labels, predictions)
	f1_class_1 = f1_score(labels, predictions, pos_label=1)
	results['F1 Class 1'] = f1_class_1
	f1_class_2 = f1_score(labels, predictions, pos_label=2)
	results['F1 Class 2'] = f1_class_2
	results['Macro F1'] = (f1_class_1 + f1_class_2) / 2
	return results

def classifier_grid_search(sentences: List[str], labels: np.ndarray,
	                       classifier_name: str, budget: int,
                           oversample: bool, undersample: bool, 
						   save_results: Optional[Path] = None,
						   save_model: Optional[Path] = None, cv: int = 10,
						   n_jobs: int = -1
						   ) -> Tuple[float, float, Dict[str, Any], 
						              RandomizedSearchCV]:
	'''
	:param sentences: Sentences that are the input to the machine learning 
	                  classifier pipeline. These can be pre-tokenised and then 
					  joined on token boundary sentences if you wish to use 
					  sentences that are not just tokenised on whitespace.
	:param labels: The labels associated to the sentences.
	:param classifier_name: Name associated to the classifier to use. This 
	                        should be the function names of the classifiers 
							within :py:mod:`pea_classification.classifiers`
	:param budget: Number of parameters to try within the 
	'''
	valid_classifier_names = ['complement_naive_bayes', 'multinomial_naive_bayes',
							  'random_forest', 'svm_classifier', 'mlp_classifier']
	if classifier_name not in valid_classifier_names:
		raise ValueError('Classifier name has to be one of the following: '
						 f'{valid_classifier_names}\nand not {classifier_name}')
	classifier = getattr(classifiers, classifier_name)()
	search_features = classifiers.pipeline_search_features(classifier_name)
	pipeline = classifiers.classifier_pipeline(oversample=oversample, 
	                                           undersample=undersample, 
											   classifier=classifier)
	score_dict = {'accuracy': make_scorer(accuracy_score), 
	              'f1 class 1': make_scorer(f1_score, pos_label=1),
				  'f1 class 2': make_scorer(f1_score, pos_label=2)}
	search_pipeline = RandomizedSearchCV(pipeline, 
	                                     param_distributions=search_features, 
										 refit='accuracy', scoring=score_dict, 
										 n_jobs=n_jobs, n_iter=budget, cv=cv, 
										 error_score=0, random_state=42)
	search_pipeline.fit(sentences, labels)
	results_df = pd.DataFrame(search_pipeline.cv_results_)
	sorted_result_df = results_df.sort_values('mean_test_accuracy').copy()
	best_mean_acc = sorted_result_df.loc[:, 'mean_test_accuracy'].iloc[-1]
	best_mean_sd = sorted_result_df.loc[:, 'std_test_accuracy'].iloc[-1]
	
	if save_results is not None:
		save_results.parent.mkdir(parents=True, exist_ok=True)
		results_df.to_csv(save_results)
	if save_model is not None:
		save_model.parent.mkdir(parents=True, exist_ok=True)
		joblib.dump(search_pipeline, str(save_model.resolve()))

	return best_mean_acc, best_mean_sd, search_pipeline.best_params_, search_pipeline

def word_list_cv_scoring(sentences: List[str], labels: np.ndarray,
	                     word_list_name: str, save_results: Optional[Path] = None,
						 save_model: Optional[Path] = None, cv: int = 10,
						 n_jobs: Optional[int] = None, 
						 **word_list_classifier_kwargs
						 ) -> Tuple[float, float, WordListClassifier]:
	score_dict = {'accuracy': make_scorer(accuracy_score), 
	              'f1 class 1': make_scorer(f1_score, pos_label=1),
				  'f1 class 2': make_scorer(f1_score, pos_label=2)}
	word_list_classifier = WordListClassifier(word_list_name, n_jobs=n_jobs, 
	                                          **word_list_classifier_kwargs)
	scores = cross_validate(word_list_classifier, sentences, labels, 
                            scoring=score_dict, cv=cv)
	scores_acc = scores['test_accuracy']
	mean_acc = scores_acc.mean()
	std_acc = scores_acc.std()
    
	f1_class_1 = scores['test_f1 class 1']
	mean_f1_class_1 = f1_class_1.mean()
	std_f1_class_1 = f1_class_1.std()

	f1_class_2 = scores['test_f1 class 2']
	mean_f1_class_2 = f1_class_2.mean()
	std_f1_class_2 = f1_class_2.std()

	results_dict = {'mean_test_f1 class 1': [mean_f1_class_1], 
	                'mean_test_f1 class 2': [mean_f1_class_2],
					'std_test_f1 class 1': [std_f1_class_1],
					'std_test_f1 class 2': [std_f1_class_2],
					'mean_test_accuracy': [mean_acc],
					'std_test_accuracy': [std_acc]}
	results_df = pd.DataFrame(results_dict) 

	# The fitting does not require any sentences it just ensures 
    # the word lists are downloaded.
	word_list_classifier.fit(sentences, labels)
	if save_results is not None:
		save_results.parent.mkdir(parents=True, exist_ok=True)
		results_df.to_csv(save_results)
	if save_model is not None:
		save_model.parent.mkdir(parents=True, exist_ok=True)
		joblib.dump(word_list_classifier, str(save_model.resolve()))

	return mean_acc, std_acc, word_list_classifier

def get_pre_trained_models(model_name: str, sampling_name: str, 
                           experiment_id: str, metric: str = 'accuracy') -> BaseEstimator:
	'''
	:param model_name: Name of the machine learning classifier or word list 
	                   e.g. `L&M` or `svm_classifier`
	:param sampling_name: either `over_sampled`, `un_balanced`, or 
	                      `under_sampled`. This is the dataset balancing 
						  strategy used when training the method/classifier.
						  Note this does not matter for the word lists as 
						  the word lists are not trained, but this 
						  argument is still required.
	:param experiment_id: The experiment that the model was trained on.
	:param metric: Will return the best tuned model for this metric.
	:returns: The loaded pre-trained model given the experiment it was trained 
	          on, the balancing/sampling strategy used to train the model, and 
			  the metric it performed best on e.g. accuracy would return the 
			  machine learning model that tunned best for accuracy, which might 
			  be different to the one that performed best on macro f1.
	'''
	possible_experiment_ids = ['exp1', 'exp2', 'exp3', 'exp4', 'exp5', 'exp6a', 
	                           'exp6b', 'exp8', 'exp9', 'exp10a', 'exp10b']
	if experiment_id in ['7a', '7b']:
		raise ValueError('Even though there are classifiers for these they are'
	                     ' the same as 6a and 6b and would be better if '
						 'you use those to stop duplication.')
	if experiment_id in ['11a', '11b']:
		raise ValueError('Even though there are classifiers for these they are'
	                     ' the same as 10a and 10b and would be better if '
						 'you use those to stop duplication.')
	if experiment_id not in possible_experiment_ids:
		raise ValueError(f'The given experiment_id {experiment_id} is not in '
	                     f'the list of possible ids {possible_experiment_ids}')

	possible_sampling_names = ['over_sampled', 'un_balanced', 'under_sampled']
	if sampling_name not in possible_sampling_names:
		raise ValueError(f'sampling name given {sampling_name} is not in the '
	                     f'list of possible names {possible_sampling_names}')
	metric_list = ['accuracy', 'macro_f1']
	if metric not in metric_list:
		raise ValueError(f'metric has to be in the following list {metric_list}'
	                     f' and not {metric}')
    
	# Download model or get from cache
	cache_dir = Path(Path.home(), '.pea_classification').resolve()
	base_url = 'https://delta.lancs.ac.uk/cfie/pea_classification_zoo/raw/master/models/'
	base_url += f'{metric}/{experiment_id}/{sampling_name}/{model_name}.joblib'
	model_fp = _get_from_cache(base_url, cache_dir)
	return joblib.load(model_fp)

def evaluate_methods(sentences: Union[List[str], Dict[str, List[str]]], 
                     labels: np.ndarray, 
					 methods: List[Union[BaseEstimator, Pipeline]],
					 method_names: List[str]
					 ) -> pd.DataFrame:
	'''
	:param sentences: Either a list of sentences or a dictionary contaning the 
	                   key `sentence` with a list of sentences and the rest of 
					   the keys are unique sentence identifiers where the values 
					   for each key are lists of same length as the sentences 
					   values.
	:param labels: The True labels associated to the sentences.
	:param methods: A list of fitted/trained machine learning methods and or 
	                word list classifiers. For the machine learning classifiers 
					this probably should be a pipeline so that it can 
					convert the text into features.
	:param method_names: A list of method names to associate to each method in 
	                     the `methods` argument
	:returns: Each method is a column and each value in the column is either 1 
	          if it got the sentence correct or 0. If the sentences argument 
			  was given as a dictionary the index of the dataframe will be the 
			  sentence identifiers.
	'''
	# Dictionary of the method name and a list of length labels stating if they 
	# get the prediction on the sentence correct.
	method_correct = {}
	_ids = None
	if isinstance(sentences, dict):
		temp_sentences = sentences.pop('sentence')
		_ids = {column_name: values for column_name, values in sentences.items()}
		sentences = temp_sentences

	for method, method_name in zip(methods, method_names):
		predictions = method.predict(sentences)
		correct = (predictions == labels).astype(int)
		method_correct[method_name] = correct
	
	if _ids is not None:
		method_correct = {**method_correct, **_ids}
		method_correct = pd.DataFrame(method_correct)
		_id_column_names = list(_ids.keys())
		alt_id_cols = ['DocumentID', 'SentenceID', 'SentmainID']
		if len(set(_id_column_names).union(set(alt_id_cols))) == len(set(_id_column_names)):
			method_correct = method_correct.set_index(alt_id_cols)
			col_order = method_names
			current_cols = method_correct.columns.tolist()
			if len(col_order) != len(current_cols):
				set_current_cols = set(current_cols)
				set_col_order = set(col_order)
				other_cols = set_current_cols.difference(set_col_order)
				col_order = list(other_cols) + col_order
			return method_correct[col_order]
		else:
			method_correct = method_correct.set_index(_id_column_names)
			return method_correct
	else:
		return pd.DataFrame(method_correct)

	