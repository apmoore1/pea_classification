# PEA Classification
Preliminary Earnings Announcements (PEA) dataset ([El-Haj et al. 2016](https://www.aclweb.org/anthology/L16-1287.pdf)) consists of sentences from multiple random PEA documents from British firms. Each sentence is annotated with the following annotations:
1. If the sentence is talking about the performance of the company, this makes the sentence a performance sentence.
2. The sentence contains attribution internally, externally or not at all. Here attribution means whether something within the sentence has been done because of internal or external factors. e.g. This sentence contain internal attribution "our continuing focus on tight cost control."
3. The overall tone/sentiment of the sentence.

This repository contains:
1. Code to load the dataset.
2. Code for Bag Of Words (BOW) methods.
3. Code to run word list methods e.g. Loughran and McDonald lexicon count based approach.
4. Code that runs both the BOW and word list methods across multiple different experiments from this dataset.
5. Models for the trained BOW methods and word list approaches.
6. A [notebook tutorial](./notebooks/PEA_Data_Analysis.ipynb) exploring/loading the dataset, how to run BOW and word list methods, and how to run the pre-trained BOW methods as well as word list methods.

## Requirements
1. Only been tested on Python 3.6.1
2. `pip install -r requirements.txt`

## Data
The training and test can can be automatically downloaded through this code base of which this is shown how in the tutorial and below:
``` python
import pea_classification
train_data = pea_classification.dataset_util.train_dataset()
test_data = pea_classification.dataset_util.test_dataset()
```
For those interested the dataset can also be found [here](https://delta.lancs.ac.uk/cfie/pea_classification_zoo/blob/master/data/pea_training_test_data.xls) in excel format. The dataset has the following annotations:
1. Performance -- Yes/No
2. Attribution -- No or Internal, External, Both, and Unsure
3. Tone/Sentiment -- Positive, Mixed, Neutral, Negative.

These annotations are some what simplified from what they really are in the data but cover the general meaning of the annotations. Details of the complete list of annotations can be found in the [notebook tutorial.](./notebooks/PEA_Data_Analysis.ipynb)

The data also has various Experiment columns where each column defines if the sentence should be in a particular experiment setup and what label it should be. In all cases the labels are either 1 or 2 thus a binary classification problem. These experiment columns are all named `exp*` where `*` can be from 1-11. The details of the different experiment setups are described below in the Experiments section.

## Experiments
In this project we are conducting 3 main experiments:
1. Predicting the tone/sentiment of a sentence.
2. Predicting if the sentence contains attribution and in some cases if the attribution is internal or external.
3. Predicting if the sentence is a performance sentence.

For each one of these experiments there are a couple of additional sub-experiments that are related, and can be found within there relevant section. For each experiment the following five machine learning methods are used:
1. Multinomial Naive Bayes
2. [Complement Naive Bayes](https://people.csail.mit.edu/jrennie/papers/icml03-nb.pdf)
3. Random Forests
4. Linear SVM
5. Multi-Layer Perceptron (MLP) Neural Network (NN)

In our experiments the machine learning methods take as input [Spacy](https://spacy.io/) tokenised text which has been lower cased. These tokens are then treated as a bag of words that are then put through a TF-IDF transformation of which the IDF function is log smoothed function. Then the top *k* TF-IDF features are chosen using the chi-squared statistic. Lastly the features are then scaled between 0-1 based on each features maximum value. These scaled features are then input into the machine learning method.

Each machine learning method will also be hyper-parameter tuned on 10 cross validation folds of the training data using a random parameter search with a budget of 40 for each experiment. The parameters that each machine learning method can change are the following:
1. Bag of word features: either uni-grams or uni-grams and bi-grams.
2. *k* when selecting the features using chi-squared statistic: *k* is chosen from a random uniform distribution between 100 and 1500.
3. scaling the features: Whether to scale the features or not.

Further more each machine learning method has there own specific parameters they can change these are:
1. Multinomial Naive Bayes
    * label/class prior: Whether or not learn the class prior or use a uniform prior.
2. Complement Naive Bayes
    * label/class prior: Whether or not learn the class prior or use a uniform prior.
    * weight normlisation: Whether or not to normalise the weights based on the document length.
3. Random Forests
    * Number of trees: Chosen from a random uniform distribution between 10 and 50.
4. Linear SVM
    * C Penalty: One of the following: [10,1,0.1,0.001,0.0001]
5. MLP NN
    * NN dimensions: One of the following: [(50,), (100, 50), (50, 25), (75, 35)]. Where (50,) means one layer NN with the first layer of size 50 and (100, 50) means two layer NN with the first layer of size 100 and the second 50.

As we have stated the machine learning methods here we describe the different word lists that we will use to classify the data as well. Unlike the machine learning methods these word lists only make sense to use in certain experiments therefore in the list we will state the experiments they will be used in:
1. [Loughran & McDonald (L&M)](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2504147) -- Sentiment/Tone experiments as well as `exp10a` and `exp10b`
2. [Henry 2008 (HEN_08)](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=933100) -- Sentiment/Tone experiments as well as `exp10a` and `exp10b`
3. [Henry 2006 (HEN_06)](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=958749) -- Sentiment/Tone experiments as well as `exp10a` and `exp10b`
4. [Zhang and Aerts 2015 (ZA_2015)](http://dx.doi.org/10.1080/00014788.2015.1048771) -- Attribution only (`exp3`, `exp4`, `exp6a`, `exp6b`)
5. [Dikoli et al. 2016 (Dikoli_2016)](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2131476) -- Attribution only (`exp3`, `exp4`, `exp6a`, `exp6b`)
6. [Martin Walker All (MW_ALL)]() -- Attribution only (`exp3`, `exp4`, `exp6a`, `exp6b`)
7. [Martin Walker 50 (MW_50)]() -- Attribution only (`exp3`, `exp4`, `exp6a`, `exp6b`)
8. [Martin Walker Attribution Type (MW_TYPE)]() -- Attribution type only (`exp5`)
Further more compared to the original word lists these come from we lower case all words and try to find the words relevant US and UK equivalent words using a dictionary derived from [tysto.com](http://www.tysto.com/uk-us-spelling-list.html). For a full list of all of these US to UK and vice versa see `pea_classification.word_list_util.US2UK`. 


Below we state the different experiments that will be conducted:

**NOTE** that the column name e.g. `exp6a` is what we classify as the unique experiment id.
### Tone/Sentiment
1. Experiment 1 (column in dataset = `exp1`): Can we predict positive from negative tone given the sentence is a performance sentence? (labels 1 = positive, 2 = negative)
2. Experiment 2 (column in dataset = `exp2`): Same as `exp1` but further conditioned on that the sentence does not contain any attribution. (labels 1 = positive, 2 = negative)

### Attribution
3. Experiment 3 (column in dataset = `exp3`): Can we predict if the sentence is an attribution sentence or not? (labels 1 = attribution, 2 = no attribution)
4. Experiment 4 (column in dataset = `exp4`): Same as `exp3` but further conditioned on that the sentence is a performance sentence. (labels 1 = attribution, 2 = no attribution)
5. Experiment 5 (column in dataset = `exp5`): Can we predict if the sentence is Internal or External attribution conditioned on the sentence being an attribution sentence. (labels 1 = Internal, 2 = External) (All case of both, unsure, and no attribution are removed for this experiment).
6. Experiment 6a (column in dataset = `exp6a`): Can we predict an attribution sentence if the attribution sentences are only **Internal** attribution sentences? (labels 1 = attribution, 2 = no attribution) (this is the same experiment setup as column `exp7a`)
7. Experiment 6b (column in dataset = `exp6b`): Can we predict an attribution sentence if the attribution sentences are only **External** attribution sentences? (labels 1 = attribution, 2 = no attribution) (this is the same experiment setup as column `exp7b`)

### Performance
8. Experiment 7 (column in the dataset = `exp8`): Can we predict performance from non-performance sentences? (label 1 = performance, 2 = non-performance)
9. Experiment 8 (column in the dataset = `exp9`): Can we predict performance from non-performance given that the sentence is not an attribution sentence? (label 1 = performance, 2 = non-performance)
10. Experiment 9 (column in the dataset = `exp10a`): Can we predict the performance sentences from the non-performance given that all performance sentences are positive in sentiment/tone? (label 1 = performance, 2 = non-performance) (this is the same experiment setup as column `exp11a`)
11. Experiment 10 (column in the dataset = `exp10b`): Can we predict the performance sentences from the non-performance given that all performance sentences are **negative** in sentiment/tone? (label 1 = performance, 2 = non-performance) (this is the same experiment setup as column `exp11b`)

As in all of these experiments the data is in-balanced we therefore perform all of the experiments in three different balancing/sampling setups:
1. No sampling just use the data as is `un_balanced`
2. Random over sample the data to the largest class size `over_sampled`
3. Random under sample the data to the smallest class size `under_sampled`

All of the training scores for each experiment including all of the scores for each hyperparameter search will be stored at the following file path given `experiment id`,  `balancing setup`, and `method name`:

`./scores/unique_experiment_id/balanced_name/method_name/train.csv`

1. unique_experiment_id -- This is the column name in the dataset e.g. for experiment 1 it is `exp1` and for experiment 7 it is `exp8`
2. balanced_name -- This can be one of three values:
    * `over_sampled` -- balance the dataset by over sampling to the largest class.
    * `under_sampled` -- balance the dataset by under sampling to the smallest class.
    * `un_balanced` -- do not balance and keep the dataset as it was.
3. method_name -- Name of one of the 5 methods used e.g. `Random_Forest` or the name of the word list used if applicable e.g. `L&M`.
4. train.csv -- This will contain the cross validation results for all of the different hyperparameters that were tried. For the word lists as they do not contain hyperparameters this will be still a CSV file.

When running these experiments we will also save two best performing models, one is the best model based on accuracy and the other on macro F1 metric. Both of thee will be saved in a [joblib](https://scikit-learn.org/stable/modules/model_persistence.html) format at the following file path (the word lists models will also be saved where applicable):

`./models/metric/unique_experiment_id/balanced_name/method_name.joblib`

1. The metric that model performed best in. This will be either `accuracy` or `macro_f1`
2. unique_experiment_id -- Name of the experiment e.g. `EXP1` for the experiment on `sentence performance sentence or not`
3. Whether the method was trained on `over_sampled`, `under_sampled`, or `un_balanced`
4. method_name -- name of the method e.g. `random_forest`

To perform this experiment run the following python script, where `./cross_val_scores` is the top level folder you want to save the scores to, `./models` is the top level folder you want to save the models to, and `./training_accuracy_aggregate_results.xls`, and `./training_macro_f1_aggregate_results.xls` is where you want the aggregate results to be stored in an excel formatted file for the best performing models based on accuracy and macro F1 respectively. Lastly the `./training_accuracy_raw_results.csv`, and `./training_macro_f1_raw_results.csv` is where the raw results are saved for the best performing accuracy and macro F1 models respectively, which can be used later to be better manipulated by [pandas](https://pandas.pydata.org/) for instance. NOTE that when running this a second time if the results and models exist for a particular method, experiment, and balancing setup it will not re-run the experiment.

``` bash
python run_experiments.py ./cross_val_scores ./models ./training_accuracy_aggregate_results.xls ./training_accuracy_raw_results.csv ./training_macro_f1_aggregate_results.xls ./training_macro_f1_raw_results.csv
```

After we have ran this function we have made available: 
* [cross_val_scores folder](https://delta.lancs.ac.uk/cfie/pea_classification_zoo/tree/master) 
* [models folder](https://delta.lancs.ac.uk/cfie/pea_classification_zoo/tree/master)
* [training_accuracy_aggregate_results.xls](https://delta.lancs.ac.uk/cfie/pea_classification_zoo/blob/master/results/training_accuracy_aggregate_results.xls) 
* [training_macro_f1_aggregate_results.xls](https://delta.lancs.ac.uk/cfie/pea_classification_zoo/blob/master/results/training_macro_f1_aggregate_results.xls)
* [training_accuracy_raw_results.csv](https://delta.lancs.ac.uk/cfie/pea_classification_zoo/blob/master/results/training_accuracy_raw_results.csv)
* [training_macro_f1_raw_results.csv](https://delta.lancs.ac.uk/cfie/pea_classification_zoo/blob/master/results/training_macro_f1_raw_results.csv)

Note that the some of the method names in the result files have been shortened (Multinomial Naive Bayes = M_BN, Complement Naive Bayes = C_NB, Random Forests = RF, Linear SVM = SVM).

Lastly we ran the best performing machine models and the word lists on the test set using the following commands. 
``` bash
python create_test_results.py test_raw_results_accuracy.csv accuracy
python create_test_results.py test_raw_results_macro_f1.csv macro_f1
python create_test_results.py test_sentence_results_accuracy.xls accuracy --sentence_level_results
python create_test_results.py test_sentence_results_macro_f1.xls macro_f1 --sentence_level_results
```

The first argument to this command e.g. `test_raw_results_accuracy.csv` is where to save the data to, `accuracy` is used to find the best model from the 10-fold cross validation based on this metric (note that for the word lists the metric does not matter as they have no learnable parameters). Finally the `--sentence_level_results` indicates how the results should be outputted. When `--sentence_level_results` this is present the results will be saved in an Excel format where each excel sheet is an experiment within these sheets each row is a test sample and each column for each method indicates if that method got it right `1` value or not `0` value. The columns `DocumentID`, `SentenceID`, and `SentmainID` are unique identifiers for those samples and `perftone`, `senttype`, and `atttype` are True annotation values.

The other output format when `--sentence_level_results` does not exist is the same as the raw results from the 10-fold training outputs. Also to NOTE with the test experiments we only used one of the Naive Bayes (NB) methods, the Complement Naive Bayes as we found this to outperform the Multinomial version in the training results.

We have also made public all of the results from the test experiments:
* [test_raw_results_accuracy.csv](https://delta.lancs.ac.uk/cfie/pea_classification_zoo/blob/master/results/test_raw_results_accuracy.csv)
* [test_raw_results_macro_f1.csv](https://delta.lancs.ac.uk/cfie/pea_classification_zoo/blob/master/results/test_raw_results_macro_f1.csv)
* [test_sentence_results_accuracy.xls](https://delta.lancs.ac.uk/cfie/pea_classification_zoo/blob/master/results/test_sentence_results_accuracy.xls)
* [test_sentence_results_macro_f1.xls](https://delta.lancs.ac.uk/cfie/pea_classification_zoo/blob/master/results/test_sentence_results_macro_f1.xls)
