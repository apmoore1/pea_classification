from typing import Union

from imblearn.over_sampling import RandomOverSampler 
from imblearn.pipeline import Pipeline as BalancePipeline
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer


def svm_classifier(random_state: Union[int, np.random.RandomState] = 42
                   ) -> SGDClassifier:
    '''
    Parameter that can be tunned is the `C` parameter.

    :param random_state: Whether or not to set the random state. See the Linear 
                         SVC `documentation <<https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html#sklearn.svm.LinearSVC>`_>`_ 
                         to know what values this can take. Default value 42.
    :returns: The default linear `Support Vector Classifier 
              <https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html#sklearn.svm.LinearSVC>`_
    '''
    return LinearSVC(random_state=random_state)

def mlp_classifier(random_state: Union[int, np.random.RandomState] = 42
                   ) -> MLPClassifier:
    '''
    Parameter that can be tunned is the `hidden_layer_sizes` parameter. 

    :param random_state: Whether or not to set the random state. See the Linear 
                         SVC `documentation <<https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html#sklearn.svm.LinearSVC>`_>`_ 
                         to know what values this can take. Default value 42.
    :returns: The default linear `Support Vector Classifier 
              <https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html#sklearn.svm.LinearSVC>`_
    '''
    return MLPClassifier(hidden_layer_sizes=(100,50), activation='relu', 
                         solver='adam', batch_size=32, learning_rate=1e-3, 
                         max_iter=200, shuffle=True, random_state=random_state, 
                         early_stopping=True, n_iter_no_change=5)

def classifier_pipeline(oversample: bool, undersample: bool) -> BalancePipeline:
    text_clf = Pipeline([('vect', CountVectorizer(ngram_range=(1,2))),
    ('clf', SGDClassifier(loss='hinge', penalty='l2',
                          alpha=1e-2, random_state=42,
                          max_iter=5, tol=None)),
])