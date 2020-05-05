import re
from typing import Any, Dict, List, Optional, Union
from multiprocessing import Pool

import numpy as np
import scipy
from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline as BalancePipeline
from imblearn.under_sampling import RandomUnderSampler
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.naive_bayes import ComplementNB, MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MaxAbsScaler
from sklearn.svm import LinearSVC

from pea_classification.word_list_util import download_and_read_word_list


def svm_classifier(random_state: Union[int, np.random.RandomState] = 42
                   ) -> LinearSVC:
    '''
    Parameter that can be tunned is the `C` parameter.

    :param random_state: Whether or not to set the random state. See the Linear 
                         SVC `documentation <https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html#sklearn.svm.LinearSVC>`_ 
                         to know what values this can take. Default value 42.
    :returns: The default linear `Support Vector Classifier 
              <https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html#sklearn.svm.LinearSVC>`_
    '''
    return LinearSVC(random_state=random_state, C=1.0)

def mlp_classifier(random_state: Union[int, np.random.RandomState] = 42
                   ) -> MLPClassifier:
    '''
    Parameter that can be tunned is the `hidden_layer_sizes` parameter. 

    :param random_state: Whether or not to set the random state. See the MLP  
                         classifier `documentation <https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html>`_ 
                         to know what values this can take. Default value 42.
    :returns: The default ` Multi-Layer Perceptron classifier
              <https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html>`_
    '''
    return MLPClassifier(hidden_layer_sizes=(100,50), activation='relu', 
                         solver='adam', batch_size=256, learning_rate_init=1e-3, 
                         max_iter=200, shuffle=True, random_state=random_state, 
                         early_stopping=True, n_iter_no_change=5)

def random_forest(random_state: Union[int, np.random.RandomState] = 42) -> RandomForestClassifier:
    '''
    Parameter that can be tunned is the `n_estimators` parameter. 

    :param random_state: Whether or not to set the random state. See the Random 
                         forest `documentation <https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html>`_ 
                         to know what values this can take. Default value 42.
    :returns: The default `Random forest classifier 
              <https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html>`_
    '''
    return RandomForestClassifier(n_estimators=10, criterion='gini', 
                                  bootstrap=True, random_state=random_state)

def multinomial_naive_bayes() -> MultinomialNB:
    '''
    Parameters that can be tunned is the `fit_prior` parameters.

    :returns: The default `Multinomial Naive Bayes 
              <https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.MultinomialNB.html#sklearn.naive_bayes.MultinomialNB>`_
    '''
    return MultinomialNB(alpha=1.0, fit_prior=True, class_prior=None)


def complement_naive_bayes() -> ComplementNB:
    '''
    Parameters that can be tunned are the `fit_prior`, and `norm` 
    parameters.

    This version of Naive Bayes is suppose to be better at modeling text 
    classification than the more traditional Multinomial Naive Bayes.
    `Reference <https://people.csail.mit.edu/jrennie/papers/icml03-nb.pdf>`_

    :returns: The default `Complement Naive Bayes 
              <https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.ComplementNB.html#sklearn.naive_bayes.ComplementNB>`_
    '''
    return ComplementNB(alpha=1.0, fit_prior=True, class_prior=None, norm=False)

def pipeline_search_features(classifier_name: str) -> Dict[str, Any]:
    '''
    Given the pipeline in `classifier_pipeline` it returns a Dictionary of 
    parameters that can be passed with the pipeline into a `RandomSearchCV` 
    model selector. Assuming that the classifier that is within the pipeline is
    the one stated in the arguments. The classifier supported are those that 
    are within this module.

    :param classifier_name: Name of the classifier that is within the
                            `classifier_pipeline` pipeline.
    :returns: A dictionary of parameters that can be passed in to a 
              `RandomSearchCV` model selector.
    '''
    default_features = {'vect__ngram_range': [(1,1), (1,2)],
                        'feature_selector__k': scipy.stats.randint(100, 1500),
                        'scale': [None, MaxAbsScaler()]}
    valid_classifier_names = ['complement_naive_bayes', 'multinomial_naive_bayes',
                              'random_forest', 'svm_classifier', 'mlp_classifier']
    if classifier_name == 'complement_naive_bayes':
        default_features['clf__fit_prior'] = [True, False]
        default_features['clf__norm'] = [True, False]
    elif classifier_name == 'multinomial_naive_bayes':
        default_features['clf__fit_prior'] = [True, False]
    elif classifier_name == 'random_forest':
        default_features['clf__n_estimators'] = scipy.stats.randint(10, 50)
    elif classifier_name == 'svm_classifier':
        default_features['clf__C'] = [10,1,0.1,0.001,0.0001]
    elif classifier_name == 'mlp_classifier':
        default_features['clf__hidden_layer_sizes'] = [(50,), (100, 50), 
                                                       (50, 25), (75, 35)]
    else:
        raise ValueError('Classifier name has to be one of the following: '
                         f'{valid_classifier_names}\nand not {classifier_name}')
    return default_features

def classifier_pipeline(oversample: bool, undersample: bool, 
                        classifier: BaseEstimator) -> BalancePipeline:
    '''
    For clarification the default pipeline here has the following linear 
    steps:
    1. CountVectorizer -- uni-grams and bi-grams feature extracted from tokens 
       that have been lower cased and split using a whitespace tokenizer.
    2. TF-IDF -- the features are then put through a TF-IDF transformer. By 
       default smooth IDF is used and the actual TF counts are used 
       (not log transformed).
    3. Feature Selection -- The features are then sub-sampled to *k* most 
       important features based on chi-squared statistics. *k* has to be 
       defined by the user else the default *k* is 10.
    4. Sampling -- Optionally random over or under sampling is performed based 
       on the arguments to this method.
    5. Scaling -- scale each feature seperately based on each features max value 
       so that all features are between 0-1.
    6. Classifier -- these features are then given to the classifier that is 
       given in the arguments to this method.

    :param oversample: Whether or not the pipeline should randomly oversample 
                       the data to the most frequent class.
    :param undersample: Whether or not the pipeline should randomly undersample 
                        the data to the least frequent class.
    :param classifier: A classifier object that can be trained to make 
                       predictions e.g. Linear SVC
    :returns: A pipeline that converts text into uni-gram and bi-gram features 
              , then performs TF-IDF on those features, then optionally over 
              or under samples and finally puts it through the classifier.
    '''
    if oversample and undersample:
        raise ValueError(f'Cannot oversample and undersample at the same time')
    default_pipeline_args = [('vect', CountVectorizer(ngram_range=(1,2), 
                                                      lowercase=True,
                                                      tokenizer=str.split)),
                             ('tfidf', TfidfTransformer(norm=None)),
                             ('feature_selector', SelectKBest(chi2)),
                             ('scale', MaxAbsScaler()),
                             ('clf', classifier)]
    if undersample:
        undersampler = RandomUnderSampler(random_state=42)
        default_pipeline_args.insert(3, ('sampler', undersampler))
    elif oversample:
        oversampler = RandomOverSampler(random_state=42)
        default_pipeline_args.insert(3, ('sampler', oversampler))
    return  BalancePipeline(default_pipeline_args)

class WordListClassifier(BaseEstimator, ClassifierMixin):
    '''
    The following word lists classify sentences by counting the number of 
    positive words compared to the number of negative words, if more positive 
    labelled positive else negative, ties or zero occurrences are negative labels:
    1. L&M, 2. HEN_08, 3. HEN_06

    If any words occur in these list then the sentence is classified as positive 
    else negative:
    1. ZA_2015, 2. Dikoli_2016, 3. MW_ALL, 4. MW_50

    If more words occur in the internal list than the external then it is 
    assigned the positive label else negative for the following list:
    1. MW_TYPE

    The following 
    :param word_list: Name of the word list to be used to classify sentences. 
                      Acceptable word list names: 1. L&M, 2. HEN_08, 3. HEN_06,
                      4. ZA_2015, 5. Dikoli_2016, 6. MW_ALL, 7. MW_50, 8. MW_TYPE
    :param pos_label: The label to assign to the positive class.
    :param neg_label: The label to assign to the negative class.
    :param n_jobs: Number of CPU's to use if left as None then all cpus will be 
                   used.
    '''
    def regex_replace_words(self, word_list: List[str], 
                            full_regex_replace: bool = False) -> List[str]:
        '''
        :param word_list: Words that when they occur in text are associated to 
                          a certain label.
        :param full_regex_replace: If `#` and `<\d:\d>\s*` words are to be replaced 
                                   with a regular expression where for `#` allows 
                                   any character to be replaced with it between 0 and 
                                   7 characters. `<\d:\d>\s*` allows between the 
                                   the first digit and second digit number of 
                                   words to be replaced with it e.g. `<1:4>` allows 
                                   between 1 and 4 words to be replaced with it.
        :returns: The word list given in arguments but with regular expression 
                  inserted in to detect word boundaries. If using 
                  `full_regex_replace` more rules are applied to detect 
                  other relevant words/word patterns.
        '''
        regext_word_list = []
        for word in word_list:
            if full_regex_replace:
                word = word.replace('#', r'\w{0,7}([\s]|$)')
                digit_search = re.findall(r'<\d:\d>\s*', word)
                for first_second in digit_search:
                    first_digit, second_digit = first_second[1], first_second[3]
                    replacement = '(\w+(\s|$)?){' + first_digit + ',' + second_digit + '}' 
                    word = re.sub(r'<\d:\d>\s*', replacement, word)
            word = r'([\s]|^)' + word + r'([\s]|$)'
            regext_word_list.append(word)
        return regext_word_list

    def count_occurrences(self, sentence: str, word_list: List[str]) -> int:
        '''
        :param sentence: A sentence.
        :param word_list: The regext word list.
        :returns: A count of the number of words from the word list that are 
                  in the given sentence.
        '''
        total_count = 0
        for word in word_list:
            total_count += len(re.findall(word, sentence))
        return total_count

    def __init__(self, word_list: str = '', pos_label: int = 1, 
                 neg_label: int = 2, n_jobs: Optional[int] = None) -> None:
        self.word_list = word_list
        self.pos_label = pos_label
        self.neg_label = neg_label
        self.n_jobs = n_jobs

    def fit(self, X: List[str], y: Optional[np.ndarray] = None) -> None:
        '''
        The fit method here mainly downloads and loads the positive and negative 
        word lists associated with the `word_list` argument specified in the 
        constructor. NOTE that it only re-downloads the word lists if the 
        word list is not already downloaded.

        :param X: A list of sentences
        :param y: Not required
        '''
        self._positive_words = None
        self._negative_words = None
        self._positive_regex_words = None
        self._negative_regex_words = None

        accepted_word_lists = ['L&M', 'HEN_08', 'HEN_06', 'ZA_2015', 
                               'Dikoli_2016', 'MW_ALL', 'MW_50', 'MW_TYPE']
        if self.word_list == 'L&M':
            self._positive_words = download_and_read_word_list('L&M pos')
            self._positive_regex_words = self.regex_replace_words(self._positive_words, False)
            self._negative_words = download_and_read_word_list('L&M neg')
            self._negative_regex_words = self.regex_replace_words(self._negative_words, False)
        elif self.word_list == 'HEN_08':
            self._positive_words = download_and_read_word_list('HEN 08 pos')
            self._positive_regex_words = self.regex_replace_words(self._positive_words, False)
            self._negative_words = download_and_read_word_list('HEN 08 neg')
            self._negative_regex_words = self.regex_replace_words(self._negative_words, False)
        elif self.word_list == 'HEN_06':
            self._positive_words = download_and_read_word_list('HEN 06 pos')
            self._positive_regex_words = self.regex_replace_words(self._positive_words, False)
            self._negative_words = download_and_read_word_list('HEN 06 neg')
            self._negative_regex_words = self.regex_replace_words(self._negative_words, False)
        elif self.word_list == 'MW_TYPE':
            self._positive_words = download_and_read_word_list('MW TYPE INT')
            self._positive_regex_words = self.regex_replace_words(self._positive_words, True)
            self._negative_words = download_and_read_word_list('MW TYPE EXT')
            self._negative_regex_words = self.regex_replace_words(self._negative_words, True)
        elif self.word_list == 'ZA_2015':
            self._positive_words = download_and_read_word_list('ZA_2015')
            self._positive_regex_words = self.regex_replace_words(self._positive_words, True)
        elif self.word_list == 'Dikoli_2016':
            self._positive_words = download_and_read_word_list('Dikoli_2016')
            self._positive_regex_words = self.regex_replace_words(self._positive_words, True)
        elif self.word_list == 'MW_ALL':
            self._positive_words = download_and_read_word_list('MW_ALL')
            self._positive_regex_words = self.regex_replace_words(self._positive_words, True)
        elif self.word_list == 'MW_50':
            self._positive_words = download_and_read_word_list('MW_50')
            self._positive_regex_words = self.regex_replace_words(self._positive_words, True)
        else:
            raise ValueError('Do not recognise this word list name '
                             f'{self.word_list}. Has to be one of the '
                             f'following {accepted_word_lists}')

    def predict(self, X: List[str], y=None) -> List[int]:
        '''
        :param X: A list of sentences
        :param y: Not required
        :returns: The predicted labels based on the number of occurrences of the 
                  positive and negative words occur in the given sentences.
                  See the constructor documentation on more details on 
                  classification for different word lists.
        '''
        sentiment_lists = ['L&M', 'HEN_08', 'HEN_06']
        attribute_type = ['MW_TYPE']
        attribution = ['ZA_2015', 'Dikoli_2016', 'MW_ALL', 'MW_50']
        with Pool(processes=self.n_jobs) as pool:
            if self.word_list in sentiment_lists or self.word_list in attribute_type:
                pos_input = ((sentence, self._positive_regex_words) for sentence in X)
                neg_input = ((sentence, self._negative_regex_words) for sentence in X)
                positive_counts = pool.starmap(self.count_occurrences, pos_input)
                positive_counts = np.array(positive_counts)
                negative_counts = pool.starmap(self.count_occurrences, neg_input)
                negative_counts = np.array(negative_counts)
                
                true_pos_false_neg = (positive_counts - negative_counts) > 0
                label_func = np.vectorize(lambda x: self.pos_label if x else self.neg_label)
                return label_func(true_pos_false_neg)
            elif self.word_list in attribution:
                negative_word_err = ('The attribution or sentiment specific word'
                                     ' lists do not have negative words')
                assert self._negative_words is None, negative_word_err
                assert self._negative_regex_words is None, negative_word_err

                pos_input = ((sentence, self._positive_regex_words) for sentence in X)
                positive_counts = pool.starmap(self.count_occurrences, pos_input)
                positive_counts = np.array(positive_counts)
                true_pos = positive_counts > 0
                label_func = np.vectorize(lambda x: self.pos_label if x else self.neg_label)
                return label_func(true_pos)
            else:
                raise ValueError(f'This word list {self.word_list} is not '
                                f'associated to any classification task. Tasks '
                                f'and word list names are the following. '
                                f'tone/sentiment {sentiment_lists}. attribution '
                                f'{attribution}. attribution type {attribute_type}')
