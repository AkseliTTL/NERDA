"""
This section covers functionality for computing predictions
with a [NERDA.models.NERDA][] model.
"""

from re import S
from NERDA.preprocessing import create_dataloader
import torch
import numpy as np
from tqdm import tqdm 
from nltk.tokenize import sent_tokenize, word_tokenize
from typing import List, Callable
import transformers
import sklearn.preprocessing

def sigmoid_transform(x):
    prob = 1/(1 + np.exp(-x))
    return prob

def predict(network: torch.nn.Module, 
            sentences: List[List[str]],
            transformer_tokenizer: transformers.PreTrainedTokenizer,
            transformer_config: transformers.PretrainedConfig,
            max_len: int,
            device: str,
            tag_encoder: sklearn.preprocessing.LabelEncoder,
            tag_outside: str,
            batch_size: int = 8,
            num_workers: int = 1,
            return_tensors: bool = False,
            return_confidence: bool = False,
            pad_sequences: bool = True) -> List[List[str]]:
    """Compute predictions.

    Computes predictions for a list with word-tokenized sentences 
    with a `NERDA` model.

    Args:
        network (torch.nn.Module): Network.
        sentences (List[List[str]]): List of lists with word-tokenized
            sentences.
        transformer_tokenizer (transformers.PreTrainedTokenizer): 
            tokenizer for transformer model.
        transformer_config (transformers.PretrainedConfig): config
            for transformer model.
        max_len (int): Maximum length of sentence after applying 
            transformer tokenizer.
        device (str): Computational device.
        tag_encoder (sklearn.preprocessing.LabelEncoder): Encoder
            for Named-Entity tags.
        tag_outside (str): Special 'outside' NER tag.
        batch_size (int, optional): Batch Size for DataLoader. 
            Defaults to 8.
        num_workers (int, optional): Number of workers. Defaults
            to 1.
        return_tensors (bool, optional): if True, return tensors.
        return_confidence (bool, optional): if True, return
            confidence scores for all predicted tokens. Defaults
            to False.
        pad_sequences (bool, optional): if True, pad sequences. 
            Defaults to True.

    Returns:
        List[List[str]]: List of lists with predicted Entity
        tags.
    """
    # make sure, that input has the correct format. 
    assert isinstance(sentences, list), "'sentences' must be a list of list of word-tokens"
    assert isinstance(sentences[0], list), "'sentences' must be a list of list of word-tokens"
    assert isinstance(sentences[0][0], str), "'sentences' must be a list of list of word-tokens"
    
    # set network to appropriate mode.
    network.eval()

    # fill 'dummy' tags (expected input for dataloader).
    tag_fill = [tag_encoder.classes_[0]]
    tags_dummy = [tag_fill * len(sent) for sent in sentences]
    dl = create_dataloader(sentences = sentences,
                           tags = tags_dummy, 
                           transformer_tokenizer = transformer_tokenizer,
                           transformer_config = transformer_config,
                           max_len = max_len, 
                           batch_size = batch_size, 
                           tag_encoder = tag_encoder,
                           tag_outside = tag_outside,
                           num_workers = num_workers,
                           pad_sequences = pad_sequences)

    predictions = []
    predictions_all = []
    probabilities = []
    tensors = []
    
    with torch.no_grad():
        for _, dl in enumerate(dl): 

            outputs = network(**dl)   

            # conduct operations on sentence level.
            for i in range(outputs.shape[0]):
                
                # extract prediction and transform.

                # find max by row.
                values, indices = outputs[i].max(dim=1)
                
                preds = tag_encoder.inverse_transform(indices.cpu().numpy())
                probs = values.cpu().numpy()

                todennäköisyydet = outputs[i].cpu().numpy()



                if return_tensors:
                    predictions_all.append(preds)   

                if return_confidence:
                    predictions_all.append(preds)

                # subset predictions for original word tokens.
                preds = [prediction for prediction, offset in zip(preds.tolist(), dl.get('offsets')[i]) if offset]
                if return_confidence:
                    probs = [prob for prob, offset in zip(probs.tolist(), dl.get('offsets')[i]) if offset]
                if return_tensors:
                    probs = [prob for prob, offset in zip(todennäköisyydet.tolist(), dl.get('offsets')[i]) if offset]
            
                # Remove special tokens ('CLS' + 'SEP').
                preds = preds[1:-1]
                if return_confidence:
                    probs = probs[1:-1]
                if return_tensors:
                    probs = probs[1:-1]
                # make sure resulting predictions have same length as
                # original sentence.
            
                # TODO: Move assert statement to unit tests. Does not work 
                # in boundary.
                # assert len(preds) == len(sentences[i])            
                predictions.append(preds)
                if return_confidence:
                    probabilities.append(np.argmax(probs, axis=-1))
                if return_tensors:
                    probabilities.append(probs)
            
    if return_confidence:
        return predictions_all, probabilities

    if return_tensors:
        return predictions, probabilities

    return predictions

def predict_arrays(network: torch.nn.Module, 
                 sentences: List[List[str]],
                 transformer_tokenizer: transformers.PreTrainedTokenizer,
                 transformer_config: transformers.PretrainedConfig,
                 max_len: int,
                 device: str,
                 tag_encoder: sklearn.preprocessing.LabelEncoder,
                 tag_outside: str,
                 batch_size: int = 8,
                 num_workers: int = 1,
                 pad_sequences: bool = True,
                 return_confidence: bool = False,
                 return_tensors: bool = False,
                 sent_tokenize: Callable = sent_tokenize,
                 word_tokenize: Callable = word_tokenize) -> tuple:

    a = [sent_tokenize(sentence) for sentence in sentences]
    part_lens = [len(i) for i in a]
    
    flat_list = [item for sublist in a for item in sublist]
    sentences = [word_tokenize(sentence) for sentence in flat_list]
    output = [' '.join(flat_list[i:i+e]) if e > 1 else flat_list[i] for i, e in enumerate(part_lens)]
    
    predictions = predict(network = network, 
                          sentences = sentences,
                          transformer_tokenizer = transformer_tokenizer,
                          transformer_config = transformer_config,
                          max_len = max_len,
                          device = device,
                          return_confidence = return_confidence,
                          batch_size = batch_size,
                          num_workers = num_workers,
                          pad_sequences = pad_sequences,
                          tag_encoder = tag_encoder,
                          tag_outside = tag_outside)

    sent_lens = [len(s) for s in sentences]
    flat_list = [item for sublist in predictions for item in sublist]
    #predictions = [' '.join(flat_list[i+sent_lens[i-1]:i+e]) if e > 1 and i > 0 else ' '.join(flat_list[i:i+e]) if e > 1 and i == 0 else flat_list[i] for i, e in enumerate(sent_lens)]
    last = 0
    num_of = 0
    counter = 0
    final = []
    for row in part_lens:
        for i in range(row):
            #print(counter+i, ' | ', counter, i)
            num_of += sent_lens[counter]
            counter += 1
        final.append(' '.join(flat_list[last:counter]))
        last = counter

    return output, final

def predict_text(network: torch.nn.Module, 
                 text: str,
                 transformer_tokenizer: transformers.PreTrainedTokenizer,
                 transformer_config: transformers.PretrainedConfig,
                 max_len: int,
                 device: str,
                 tag_encoder: sklearn.preprocessing.LabelEncoder,
                 tag_outside: str,
                 batch_size: int = 8,
                 num_workers: int = 1,
                 pad_sequences: bool = True,
                 return_confidence: bool = False,
                 return_tensors: bool = False,
                 sent_tokenize: Callable = sent_tokenize,
                 word_tokenize: Callable = word_tokenize) -> tuple:
    """Compute Predictions for Text.

    Computes predictions for a text with `NERDA` model. 
    Text is tokenized into sentences before computing predictions.

    Args:
        network (torch.nn.Module): Network.
        text (str): text to predict entities in.
        transformer_tokenizer (transformers.PreTrainedTokenizer): 
            tokenizer for transformer model.
        transformer_config (transformers.PretrainedConfig): config
            for transformer model.
        max_len (int): Maximum length of sentence after applying 
            transformer tokenizer.
        device (str): Computational device.
        tag_encoder (sklearn.preprocessing.LabelEncoder): Encoder
            for Named-Entity tags.
        tag_outside (str): Special 'outside' NER tag.
        batch_size (int, optional): Batch Size for DataLoader. 
            Defaults to 8.
        num_workers (int, optional): Number of workers. Defaults
            to 1.
        pad_sequences (bool, optional): if True, pad sequences. 
            Defaults to True.
        return_confidence (bool, optional): if True, return 
            confidence scores for predicted tokens. Defaults
            to False.

    Returns:
        tuple: sentence- and word-tokenized text with corresponding
        predicted named-entity tags.
    """
    assert isinstance(text, str), "'text' must be a string."
    sentences = sent_tokenize(text)

    sentences = [word_tokenize(sentence) for sentence in sentences]

    if return_tensors:
        predictions, probs = predict(network = network, 
                          sentences = sentences,
                          transformer_tokenizer = transformer_tokenizer,
                          transformer_config = transformer_config,
                          max_len = max_len,
                          device = device,
                          return_confidence = return_confidence,
                          batch_size = batch_size,
                          num_workers = num_workers,
                          pad_sequences = pad_sequences,
                          return_tensors = return_tensors,
                          tag_encoder = tag_encoder,
                          tag_outside = tag_outside)
        return sentences, predictions, probs

    predictions = predict(network = network, 
                          sentences = sentences,
                          transformer_tokenizer = transformer_tokenizer,
                          transformer_config = transformer_config,
                          max_len = max_len,
                          device = device,
                          return_confidence = return_confidence,
                          batch_size = batch_size,
                          num_workers = num_workers,
                          pad_sequences = pad_sequences,
                          tag_encoder = tag_encoder,
                          tag_outside = tag_outside)

    return sentences, predictions

