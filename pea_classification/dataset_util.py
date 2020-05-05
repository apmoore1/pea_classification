from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
from urllib.parse import urlparse
import tempfile
from typing import List, Dict, Tuple, Optional, Union, IO, Callable
import shutil

import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
import spacy

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

def spacy_tokenizer() -> Callable[[str], List[str]]:
    '''
    :returns: A callable that when given text will return a list of strings 
              representing the tokens from the original text. These tokens 
              will have been tokenised by the English Spacy tokeniser.
    '''
    tok = spacy.blank('en')
    def _spacy_token_to_text(text: str) -> List[str]:
        return [spacy_token.text for spacy_token in tok(text) 
                if not spacy_token.is_space]
    return _spacy_token_to_text

def get_experiment_data(data: pd.DataFrame, experiment_id: str, 
                        sentence_column: str, return_ids: bool = False
                        ) -> Union[Tuple[List[str], np.ndarray],
                                   Tuple[List[str], np.ndarray,
                                         Dict[str, List[int]]]]:
    '''
    :param data: Data that contains a column with 1, 2 or empty values where the 
                 empty value rows are to be removed and the rest to be extracted.
    :param experiment_id: Name of the column that contain the 1, 2 or empty 
                          values.
    :param sentence_column: Name of the sentence column to get the sentence 
                            data from.
    :param return_ids: Whether or not to return the sentence ID columns.
    :returns: A Tuple of 2 values: 1. A list of sentences, 2. An array of 1 and 
              2 values, the sentences are associated to the 1 or 2 values that 
              can be used to train a classifier to predict the 1 or 2 values 
              from the associated sentence. If return_ids is True then 
              the Tuple will have an extra value a dictionary of ids and values 
              where the values are the same length as the number of sentences 
              and each value across the column names creates a unique ID for 
              the sentence and labels.
    '''
    experiment_data = data.loc[data[experiment_id].isin([1,2])].copy()
    sentences = experiment_data.loc[:, f'{sentence_column}'].tolist()
    labels = experiment_data.loc[:, experiment_id].to_numpy()
    
    shape_err = (f'The number of sentences should {len(sentences)}, should equal '
                 f'the number of labels {labels.shape[0]}')
    assert len(sentences) == labels.shape[0], shape_err

    document_id: List[int] = experiment_data.loc[:, 'DocumentID'].tolist()
    sentence_id: List[int] = experiment_data.loc[:, 'SentenceID'].tolist()
    sentmain_id: List[int] = experiment_data.loc[:, 'SentmainID'].tolist()
    assert len(document_id) == labels.shape[0]
    assert len(sentence_id) == labels.shape[0]
    assert len(sentmain_id) == labels.shape[0]
    _ids = {'DocumentID': document_id, 'SentenceID': sentence_id, 
            'SentmainID': sentmain_id}
    if return_ids:
        return sentences, labels, _ids
    else:
        return sentences, labels

#
# Code to get the train and test datasets
#
CACHE_DIRECTORY = Path(Path.home(), '.pea_classification').resolve()
DATASET_URL = 'https://delta.lancs.ac.uk/cfie/pea_classification_zoo/raw/master/data/pea_training_test_data.xls?inline=false'

def _url_to_filename(url: str, etag: str = None) -> str:
    """
    Convert `url` into a hashed filename in a repeatable way.
    If `etag` is specified, append its hash to the url's, delimited
    by a period.

    Taken from Allennlp library:
    https://github.com/allenai/allennlp/blob/master/allennlp/common/file_utils.py#L44
    """
    url_bytes = url.encode("utf-8")
    url_hash = sha256(url_bytes)
    filename = url_hash.hexdigest()

    if etag:
        etag_bytes = etag.encode("utf-8")
        etag_hash = sha256(etag_bytes)
        filename += "." + etag_hash.hexdigest()

    return filename

def _session_with_backoff() -> requests.Session:
    """
    We ran into an issue where http requests to s3 were timing out,
    possibly because we were making too many requests too quickly.
    This helper function returns a requests session that has retry-with-backoff
    built in.
    see stackoverflow.com/questions/23267409/how-to-implement-retry-mechanism-into-python-requests-library
    
    Taken from Allennlp library:
    https://github.com/allenai/allennlp/blob/master/allennlp/common/file_utils.py#L188
    """
    session = requests.Session()
    retries = Retry(total=5, backoff_factor=1, status_forcelist=[502, 503, 504])
    session.mount("http://", HTTPAdapter(max_retries=retries))
    session.mount("https://", HTTPAdapter(max_retries=retries))

    return session

def _http_get(url: str, temp_file: IO) -> None:
    '''
    Taken from Allennlp library:
    https://github.com/allenai/allennlp/blob/master/allennlp/common/file_utils.py#L204
    '''
    with _session_with_backoff() as session:
        req = session.get(url, stream=True)
        for chunk in req.iter_content(chunk_size=1024):
            if chunk:  # filter out keep-alive new chunks
                temp_file.write(chunk)

def _get_from_cache(url: str, cache_dir: Union[str, Path]) -> Path:
    '''
    Adapted from Allennlp library:
    https://github.com/allenai/allennlp/blob/master/allennlp/common/file_utils.py#L218
    '''
    with _session_with_backoff() as session:
        response = session.head(url, allow_redirects=True)
        if response.status_code != 200:
            io_error_msg = (f'HEAD request failed for url {url} with status '
                            f'code {response.status_code}')
            raise IOError(io_error_msg)
        etag = response.headers.get("ETag")
    filename = _url_to_filename(url, etag)

    # get cache path to put the file
    cache_path = Path(cache_dir, filename)

    if not cache_path.exists():
        # Download to temporary file, then copy to cache dir once finished.
        # Otherwise you get corrupt cache entries if the download gets interrupted.
        with tempfile.NamedTemporaryFile() as temp_file:
            _http_get(url, temp_file)

            # we are copying the file before closing it, so flush to avoid truncation
            temp_file.flush()
            # shutil.copyfileobj() starts at the current position, so go to the start
            temp_file.seek(0)
            with cache_path.open('wb') as cache_file:
                shutil.copyfileobj(temp_file, cache_file)

            meta = {"url": url, "etag": etag}
            meta_path = cache_path.with_suffix(cache_path.suffix + '.json')
            with meta_path.open('w') as meta_file:
                json.dump(meta, meta_file)
    return cache_path.resolve()

def get_dataset(url_or_filename: Optional[Union[Path, str]] = None, 
                cache_dir: str = None
                ) -> pd.DataFrame:
    '''
    Adapted from Allennlp library:
    https://github.com/allenai/allennlp/blob/master/allennlp/common/file_utils.py#L86

    :param url_or_filename: local file path as a string or Path object or a  
                            URL address to the data that will be downloaded into 
                            cache_dir. If not given it will download the dataset 
                            from the official university hosted version of the 
                            dataset which can be found here:
                            https://delta.lancs.ac.uk/mooreap/pea_classification_zoo/tree/master/data
    :param cache_dir: The directory to store the downloaded file if a download 
                      is required. If this is None and a download is required 
                      the default cache directory is 
                      `USER_HOME_PATH/.pea_classification`
    :returns: The pea classification dataset as a pandas dataframe.
    '''
    path_cache_dir: Path
    if cache_dir is None:
        path_cache_dir = CACHE_DIRECTORY
    else:
        path_cache_dir = Path(cache_dir).resolve()
    path_cache_dir.mkdir(parents=True, exist_ok=True)
    
    if url_or_filename is None:
        url_or_filename = DATASET_URL
    
    url_or_filename = str(url_or_filename)
    parsed = urlparse(url_or_filename)
    dataset_fp = None
    if parsed.scheme in ("http", "https"):
        # URL, so get it from the cache (downloading if necessary)
        dataset_fp = _get_from_cache(url_or_filename, path_cache_dir)
    else:
        dataset_fp = Path(url_or_filename).resolve()
        if not dataset_fp.exists():
            raise FileNotFoundError(f'This file does not exist {dataset_fp}')
    dataset = pd.read_excel(dataset_fp)
    # Remove samples from the dataset that do not contain a valid sentence
    dataset = dataset.loc[dataset['samples to remove']==0].copy()
    dataset = dataset.drop('samples to remove', axis=1)
    return dataset

def train_dataset(url_or_filename: Optional[Path] = None, 
                  cache_dir: Optional[str] = None) -> pd.DataFrame:
    '''
    :param url_or_filename: local file path as a string or Path object or a  
                            URL address to the data that will be downloaded into 
                            cache_dir. If not given it will download the dataset 
                            from the official university hosted version of the 
                            dataset which can be found here:
                            https://delta.lancs.ac.uk/mooreap/pea_classification_zoo/tree/master/data
    :param cache_dir: The directory to store the downloaded file if a download 
                      is required. If this is None and a download is required 
                      the default cache directory is 
                      `USER_HOME_PATH/.pea_classification`
    :returns: The PEA classification training dataset
    '''
    whole_dataset_df = get_dataset(url_or_filename, cache_dir)
    train_data = whole_dataset_df.loc[whole_dataset_df['train0test1']==0].copy()
    train_data = train_data.drop('train0test1', axis=1)
    return train_data

def test_dataset(url_or_filename: Optional[Path] = None, 
                 cache_dir: Optional[str] = None) -> pd.DataFrame:
    '''
    :param url_or_filename: local file path as a string or Path object or a  
                            URL address to the data that will be downloaded into 
                            cache_dir. If not given it will download the dataset 
                            from the official university hosted version of the 
                            dataset which can be found here:
                            https://delta.lancs.ac.uk/mooreap/pea_classification_zoo/tree/master/data
    :param cache_dir: The directory to store the downloaded file if a download 
                      is required. If this is None and a download is required 
                      the default cache directory is 
                      `USER_HOME_PATH/.pea_classification`
    :returns: The PEA classification test dataset.
    '''
    whole_dataset_df = get_dataset(url_or_filename, cache_dir)
    test_data = whole_dataset_df.loc[whole_dataset_df['train0test1']==1].copy()
    test_data.loc[:, 'atttype'] = test_data.loc[:, 'atttype'].replace({'N.A.': np.nan})
    test_data = test_data.drop('train0test1', axis=1)
    return test_data

def get_cv_results() -> pd.DataFrame:
    '''
    :returns: This loads the cross validation results from the experiments 
              described in this README: 
              https://github.com/apmoore1/pea_classification Where the results 
              can be found here: https://delta.lancs.ac.uk/cfie/pea_classification_zoo/blob/master/training_raw_results.csv
              Furthermore the results are also slightly changed to remove 
              rededunt experiments like 7a, 7b, 11a, and 11b.
    '''
    experiments_to_remove = ['exp7a', 'exp7b', 'exp11a', 'exp11b']
    cv_url = 'https://delta.lancs.ac.uk/cfie/pea_classification_zoo/raw/master/training_raw_results.csv?inline=false'
    cv_results = pd.read_csv(_get_from_cache(cv_url, CACHE_DIRECTORY))
    pre_processed_results = cv_results[~cv_results.loc[:, 'Experiment ID'].isin(experiments_to_remove)].copy()
    pre_processed_results = pre_processed_results.drop('Unnamed: 0', axis=1)
    return pre_processed_results
