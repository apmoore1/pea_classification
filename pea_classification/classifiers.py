from typing import Union

from imblearn.over_sampling import RandomOverSampler 
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as BalancePipeline
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer


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
                         solver='adam', batch_size=32, learning_rate=1e-3, 
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

def classifier_pipeline(oversample: bool, undersample: bool, 
                        classifier: BaseEstimator) -> BalancePipeline:
    '''
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
    default_pipeline_args = [('vect', CountVectorizer(ngram_range=(1,2))),
                             ('tfidf', TfidfTransformer()),
                             ('clf', classifier)]
    if undersample:
        undersampler = RandomUnderSampler(random_state=42)
        default_pipeline_args.insert(2, ('sampler', undersampler))
    elif oversample:
        oversampler = RandomOverSampler(random_state=42)
        default_pipeline_args.insert(2, ('sampler', oversampler))
    return  BalancePipeline(default_pipeline_args)