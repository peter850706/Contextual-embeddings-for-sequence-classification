import argparse
import csv
import random
import sys
from pathlib import Path
import functools

import ipdb
import numpy as np
import torch
import torch.nn as nn
from box import Box
from tqdm import tqdm

from BERT.dataset import create_data_loader
from BERT.train import Model
from BERT.common.losses import CrossEntropyLoss
from BERT.common.metrics import Accuracy
from BERT.common.utils import load_pkl
from pytorch_pretrained_bert import BertTokenizer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('ckpt_paths', nargs='+', type=Path, help='Model checkpoint paths')
    parser.add_argument('prediction_dir', type=Path, help='Saved predictions dir')
    parser.add_argument('--batch_size', type=int, help='Inference batch size')
    args = parser.parse_args()
    return vars(args)

def input_pad_to_len(words, padded_word_len, word_padding=0):
    """Pad words to 'padded_word_len' with padding if 'len(words) < padded_word_len'.
    Example:
        target_pad_to_len([1, 2, 3], 5, -1) == [1, 2, 3, -1, -1]
    Args:
        words (list): List of the word index.
        padded_word_len (int): The length for padding a batch of sequences to the same length. 
        word_padding (int): The index used to pad.
    """
    if len(words) < padded_word_len:
        words += [word_padding] * (padded_word_len - len(words))
    elif len(words) > padded_word_len:
        words = words[:padded_word_len]
    else:
        pass
    return words

def main(ckpt_paths, prediction_dir, batch_size):    
    models_logits = []
    for i, ckpt_path in enumerate(ckpt_paths):
        model_dir = ckpt_path.parent.parent
        try:
            cfg = Box.from_yaml(filename=model_dir / 'config.yaml')
        except FileNotFoundError:
            print('[!] Model directory({}) must contain config.yaml'.format(model_dir))
            exit(1)

        device = torch.device('{}:{}'.format(cfg.device.type, cfg.device.ordinal))
        random.seed(cfg.random_seed)
        np.random.seed(cfg.random_seed)
        torch.manual_seed(cfg.random_seed)
        torch.cuda.manual_seed_all(cfg.random_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        if i == 0:
            if not prediction_dir.exists():
                prediction_dir.mkdir()
            print('[-] Directory {} created\n'.format(prediction_dir))
            
            dataset_dir = Path(cfg.dataset_dir)
            print('[*] Loading test dataset from {}'.format(dataset_dir))
            test_dataset_path = dataset_dir / 'test.pkl'
            test_dataset = load_pkl(test_dataset_path)

            print('[*] Creating test data loader')
            if batch_size:
                cfg.data_loader.batch_size = batch_size
            test_data_loader = create_data_loader(test_dataset, **cfg.data_loader, shuffle=False)
            
        print('\n[-] Model checkpoint: {}'.format(ckpt_path))
        
        print('[*] Creating model')
        model = Model(device, cfg.net, cfg.optim, t_total=0)
        model.load_state(ckpt_path)
        
        print('[*] Creating tokenizer')
        tokenizer = BertTokenizer.from_pretrained(cfg.net.bert_pretrained_model_name)
        
        logits = get_model_logits(device, test_data_loader, model, tokenizer, mode='test')
        models_logits.append(logits)
        
    Ids, predictions = ensemble_predict(device, test_data_loader, models_logits, mode='test')
    save_predictions(Ids, predictions, prediction_dir / 'predict.csv')

def get_model_logits(device, data_loader, model, tokenizer, mode='test'):
    assert mode in ['dev', 'test']
    
    model.set_eval()
    
    if mode == 'dev':
        loss = CrossEntropyLoss(device, 'logits', 'label')
        metric = Accuracy(device, 'label')
    
    with torch.no_grad():
        bar = tqdm(data_loader, desc='[Get Model Logits]', leave=False, dynamic_ncols=True)
        logits_list = []
        for batch in bar:
            text_word = []
            for text in batch['text_orig']:
                text = ' '.join(['[CLS]'] + text)
                tokens = tokenizer.tokenize(text)
                indexed_tokens = tokenizer.convert_tokens_to_ids(tokens)
                text_word.append(indexed_tokens)

            # padding
            padded_word_len = max(map(len, text_word))
            text_word = torch.tensor(list(map(functools.partial(input_pad_to_len, 
                                                                padded_word_len=padded_word_len), 
                                              text_word)),
                                     dtype=torch.long)

            # attention mask
            zeros = torch.zeros_like(text_word)
            ones = torch.ones_like(text_word)
            attention_mask = torch.where(text_word == 0, zeros, ones)
            
            logits = model(text_word.to(device=device), attention_mask=attention_mask.to(device=device))
            logits_list.append(logits)            
            label = logits.max(dim=1)[1]
            
            if mode == 'dev':
                output = {'logits': logits, 'label': label}
                loss.update(output, batch)
                metric.update(output, batch)            
                bar.set_postfix(**{loss.name: loss.value, metric.name: metric.value})
        bar.close()
    if mode == 'dev':
        print('[-] Dev {}: {}; {}: {}\n'.format(loss.name, loss.value, metric.name, metric.value))
    return torch.cat(logits_list, dim=0)
    
def ensemble_predict(device, data_loader, models_logits, mode='test'):
    if mode == 'dev':
        loss = CrossEntropyLoss(device, 'logits', 'label')
        metric = Accuracy(device, 'label')
    
    ensemble_logits = torch.stack(models_logits, dim=0).mean(dim=0)
    ensemble_label = ensemble_logits.max(dim=1)[1]
    
    Ids = []
    predictions = []
    bar = tqdm(data_loader, desc='[Ensemble Predict]', leave=False, dynamic_ncols=True)
    batch_size = data_loader.batch_size
    for i, batch in enumerate(bar):
        Ids += batch['Id']
        logits = ensemble_logits[i * batch_size : (i + 1) * batch_size]
        label = ensemble_label[i * batch_size : (i + 1) * batch_size]
        predictions += label.tolist()
        
        if mode == 'dev':
            output = {'logits': logits, 'label': label}
            loss.update(output, batch)
            metric.update(output, batch)            
            bar.set_postfix(**{loss.name: loss.value, metric.name: metric.value})
    bar.close()
    if mode == 'dev':
        print('[-] Dev {}: {}; {}: {}\n'.format(loss.name, loss.value, metric.name, metric.value))
    return Ids, predictions

def save_predictions(Ids, predictions, output_path):
    with output_path.open(mode='w') as f:
        writer = csv.DictWriter(f, fieldnames=['Id', 'label'])
        writer.writeheader()
        writer.writerows(
            [{'Id': Id, 'label': p + 1} for Id, p in zip(Ids, predictions)])
    print('\n[-] Output saved to {}'.format(output_path))


if __name__ == "__main__":
    with ipdb.launch_ipdb_on_exception():
        sys.breakpointhook = ipdb.set_trace
        kwargs = parse_args()
        main(**kwargs)