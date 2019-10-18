from typing import Union

from imblearn.over_sampling import RandomOverSampler 
from imblearn.pipeline import Pipeline as BalancePipeline
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer


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
    return LinearSVC(random_state=random_state)

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
    Parameters that can be tunned are the `alpha`, and `fit_prior` parameters.

    :returns: The default `Multinomial Naive Bayes 
              <https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.MultinomialNB.html#sklearn.naive_bayes.MultinomialNB>`_
    '''
    return MultinomialNB(alpha=1.0, fit_prior=True, class_prior=None)


def complement_naive_bayes() -> ComplementNB:
    '''
    Parameters that can be tunned are the `alpha`, `fit_prior`, and `norm` 
    parameters.

    This version of Naive Bayes is suppose to be better at modeling text 
    classification than the more traditional Multinomial Naive Bayes.
    `Reference <https://people.csail.mit.edu/jrennie/papers/icml03-nb.pdf>`_

    :returns: The default `Complement Naive Bayes 
              <https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.ComplementNB.html#sklearn.naive_bayes.ComplementNB>`_
    '''
    return ComplementNB(alpha=1.0, fit_prior=True, class_prior=None, norm=False)

def classifier_pipeline(oversample: bool, undersample: bool) -> BalancePipeline:
    text_clf = Pipeline([('vect', CountVectorizer(ngram_range=(1,2))),
    ('clf', SGDClassifier(loss='hinge', penalty='l2',
                          alpha=1e-2, random_state=42,
                          max_iter=5, tol=None)),
])