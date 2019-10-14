# PEA Classification

## Requirements
1. Only been tested on Python 3.6
2. `pip install requirements.txt`

## Data
All of the raw training data is stored in the following excel spreadsheet `./data/training_data.xls`

Before we can use the data some data pre-processing is required:
1. Remove all samples that contain "DUPLICATE" within the "BV" column.
2. Remove all samples that contain "BLANK" within the "BV" column.

## Experiments
In this project we are conducted 4 main experiments:
1. Is a sentence a performance sentence or not.

For each one of these experiments there are a couple of additional sub-experiments that are related, and can be found within there relevant section. For each experiment including the sub-experiments the data has to be manipulated so that only the relevant sentences to be trained and tested on are given to the methods. For each experiment the following four methods are used:
1. Naive Bayes
2. Random Forests
3. SVM
4. Multi-Layer Perceptron (MLP) Neural Network (NN)

For each of theses methods the default hyper-parameters will be used from the [scikit-learn library](https://scikit-learn.org/0.21/) (version 0.21), it is suggested for the MLP NN that the [features are scaled](https://scikit-learn.org/stable/modules/neural_networks_supervised.html#tips-on-practical-use). Each of these methods will use the same features, which are uni-gram and bi-gram word features.

For each of the experiments it will have a unique identifier so that the scores can be easily extracted and plotted. All of the scores for each experiment will be saved within the `./scores` folder in the following folder structure:

`./scores/feature_name/classifier_name/unique_experiment_id/balanced_name/dataset_file`

1. feature_name -- The name of the features used. To start with this will always be `uni_bi_grams` representing the uni-gram and bi-gram word features used.
2. classifier_name -- Name of one of the 4 methods used e.g. `Random_Forest`
3. unique_experiment_id -- Name of the experiment e.g. `EXP1` for the experiment on `sentence performance sentence or not`
4. balanced_name -- This can be one of three values:
    * `over_sampled` -- balance the dataset by over sampling to the largest class.
    * `under_sampled` -- balance the dataset by under sampling to the smallest class.
    * `un_balanced` -- do not balance and keep the dataset as it was.
5. dataset_file -- This will be either `train.json` or `test.json`, where each file will contain the scores for 10 fold cross validation on the training dataset and the scores on the test set respectively.

For each experiment the dataset statistics will be created for both the training and test datasets. These statistics will be saved within the following folder structure:

`./dataset_statistics/unique_experiment_id/stats.json`

1. unique_experiment_id -- Name of the experiment e.g. `EXP1` for the experiment on `sentence performance sentence or not`
2. stats.json -- Will be the file that contains a list of tuples with three values:
    * Dataset -- `train` or `test`
    * Class Name -- name of the class e.g. `Performance Sentence`
    * Number of Sentences -- number of sentences/samples for that class in that dataset. An example value 259.
    * Normalised Number of Sentence -- number of sentences/samples for that class in that dataset divided by the total number of sentences/samples in that dataset multiplied by 100 to turn it into a percentage. An example value 25.2%.

NOTE: In these experiments each sentence represents one sample as we are performing sentence classification.

### Performance related experiments
In all of these experiments we include sentences that have neutral sentiment.

1. Is a sentence a performance sentence? (Answer Yes/No) (These include neutral sentences) UNIQUE ID: EXP1
2. Is a sentence a positive performance sentence? (Answer: Yes/No) (These do not include neutral sentences) UNIQUE ID: EXP2
3. Is a sentence a negative performance sentence? (Answer: Yes/No) (These do not include neutral sentences) UNIQUE ID: EXP3

## Saved Models
Here will be instructions on how to use the pre-trained classifiers.
