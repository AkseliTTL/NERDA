"""Precooked NERDA Models"""
from .datasets import get_dane_data
from .models import NERDA
import os
import urllib
from pathlib import Path

class Precooked(NERDA):
    """Precooked NERDA model"""
    def __init__(self, **kwargs) -> None:
        """Initialize NERDA model"""
        super().__init__(**kwargs)

    def download_network(self, dir = None):
        """Download Precooked Network"""

        model_name = type(self).__name__

        url_s3 = 'https://nerda.s3-eu-west-1.amazonaws.com'
        url_model = f'{url_s3}/{model_name}.bin'
        
        if dir is None:
            dir = os.path.join(str(Path.home()), '.nerda')

        if not os.path.exists(dir):
            os.mkdir(dir)
            
        file_path = os.path.join(dir, f'{model_name}.bin')
        
        print(f'Downloading {url_model} to {file_path}')
        urllib.request.urlretrieve(url_model, file_path)

        return "Network downloaded successfully."

    def load_network(self, file_path = None):

        model_name = type(self).__name__
        
        if file_path is None:
            file_path = os.path.join(str(Path.home()), '.nerda', f'{model_name}.bin')

        self.load_network_from_file(file_path)
        
class BERT_ML_DaNE(Precooked):
    """NERDA Multilingual BERT Finetuned on DaNE data set"""
    def __init__(self) -> None:
        """Initialize model"""
        super().__init__(transformer = 'bert-base-multilingual-uncased',
                         device = None,
                         tag_scheme = [
                            'B-PER',
                            'I-PER', 
                            'B-ORG', 
                            'I-ORG', 
                            'B-LOC', 
                            'I-LOC', 
                            'B-MISC', 
                            'I-MISC'
                            ],
                         tag_outside = 'O',
                         dataset_training = get_dane_data('train'),
                         dataset_validation = get_dane_data('dev'),
                         max_len = 128,
                         dropout = 0.1,
                         hyperparameters = {'epochs' : 4,
                                            'warmup_steps' : 500,
                                            'train_batch_size': 13,
                                            'learning_rate': 0.0001},
                         tokenizer_parameters = {'do_lower_case' : True})

class ELECTRA_DA_DaNE(Precooked):
    """NERDA Danish Electra (-l-ctra) finetuned on DaNE data set
    
    We have spent literally no time on actually finetuning the model,
    so performance can very likely be improved.
    """
    def __init__(self) -> None:
        """Initialize model"""
        super().__init__(transformer = 'Maltehb/-l-ctra-danish-electra-small-uncased',
                         device = None,
                         tag_scheme = [
                            'B-PER',
                            'I-PER', 
                            'B-ORG', 
                            'I-ORG', 
                            'B-LOC', 
                            'I-LOC', 
                            'B-MISC', 
                            'I-MISC'
                            ],
                         tag_outside = 'O',
                         dataset_training = get_dane_data('train'),
                         dataset_validation = get_dane_data('dev'),
                         max_len = 128,
                         dropout = 0.1,
                         hyperparameters = {'epochs' : 5,
                                            'warmup_steps' : 500,
                                            'train_batch_size': 13,
                                            'learning_rate': 0.0001},
                         tokenizer_parameters = {'do_lower_case' : True})