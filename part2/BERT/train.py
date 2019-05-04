import argparse
import random
import sys
from pathlib import Path
import functools
from tqdm import tqdm

import ipdb
import numpy as np
import torch
from box import Box

from BERT.dataset import create_data_loader
from BERT.common.base_model import BaseModel
from BERT.common.base_trainer import BaseTrainer
from BERT.common.losses import CrossEntropyLoss
from BERT.common.metrics import Accuracy
from BERT.common.utils import load_pkl

from pytorch_pretrained_bert import BertTokenizer
from pytorch_pretrained_bert import BertForSequenceClassification
from BERT.modules.BertForSequenceClassificationWrapper import BertForSequenceClassificationWrapper


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('model_dir', type=Path, help='Target model directory')
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

class Model(BaseModel):
    def _create_net_and_optim(self, net_cfg, optim_cfg, t_total):        
        if net_cfg.type == 'BertForSequenceClassification':
            net = BertForSequenceClassification.from_pretrained(net_cfg.bert_pretrained_model_name, 
                                                                num_labels=net_cfg.num_labels)
        elif net_cfg.type == 'BertForSequenceClassificationWrapper':
            net = BertForSequenceClassificationWrapper.from_pretrained(net_cfg.bert_pretrained_model_name, 
                                                                       num_labels=net_cfg.num_labels,
                                                                       dropout=net_cfg.dropout)
        else:
            raise ValueError
        
        if net_cfg.frozen and net_cfg.tuned_layers is not None:
            raise ValueError
        
        if net_cfg.frozen:
            for name, param in net.named_parameters():
                if 'bert' in name:
                    param.requires_grad = False
        else:
            if net_cfg.tuned_layers != 'all':
                tuned_layers = ['bert.encoder.layer.' + str(layer) for layer in net_cfg.tuned_layers]
                for name, param in net.named_parameters():
                    if 'bert' in name:
                        if not any(tuned_layer in name for tuned_layer in tuned_layers):
                            param.requires_grad = False
        
        net.to(device=self._device)
        
        net_parameters = list(filter(lambda p: p.requires_grad, net.parameters()))
        if optim_cfg.algo in dir(torch.optim):
            optim = getattr(torch.optim, optim_cfg.algo)([{'params': net_parameters}], **optim_cfg.kwargs)
        elif optim_cfg.algo == 'BertAdam': 
            from pytorch_pretrained_bert.optimization import BertAdam
            optim = BertAdam([{'params': net_parameters}], 
                             lr=optim_cfg.kwargs.lr,
                             warmup=optim_cfg.kwargs.warmup,
                             t_total=t_total)
        else:
            raise NotImplementedError
        return net, optim


class Trainer(BaseTrainer):
    def __init__(self, bert_pretrained_model_name, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tokenizer = BertTokenizer.from_pretrained(bert_pretrained_model_name)
        
    def _run_batch(self, batch):
        text_word = []
        for text in batch['text_orig']:
            text = ' '.join(['[CLS]'] + text)
            tokens = self.tokenizer.tokenize(text)
            indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokens)
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
        logits = self._model(text_word.to(device=self._device), 
                             attention_mask=attention_mask.to(device=self._device))
        label = logits.max(dim=1)[1]

        return {
            'logits': logits,
            'label': label
        }


def main(model_dir):
    try:
        cfg = Box.from_yaml(filename=model_dir / 'config.yaml')
    except FileNotFoundError:
        print('[!] Model directory({}) must contain config.yaml'.format(model_dir))
        exit(1)
    print(
        '[-] Model checkpoints and training log will be saved to {}\n'
        .format(model_dir))

    device = torch.device('{}:{}'.format(cfg.device.type, cfg.device.ordinal))
    random.seed(cfg.random_seed)
    np.random.seed(cfg.random_seed)
    torch.manual_seed(cfg.random_seed)
    torch.cuda.manual_seed_all(cfg.random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    log_path = model_dir / 'log.csv'
    ckpt_dir = model_dir / 'ckpts'
    if any([p.exists() for p in [log_path, ckpt_dir]]):
        print('[!] Directory already contains saved ckpts/log')
        exit(1)
    ckpt_dir.mkdir()

    print('[*] Loading vocabs and datasets from {}'.format(cfg.dataset_dir))
    dataset_dir = Path(cfg.dataset_dir)
    word_vocab = load_pkl(dataset_dir / 'word.pkl')
    char_vocab = load_pkl(dataset_dir / 'char.pkl')
    train_dataset = load_pkl(dataset_dir / 'train.pkl')
    dev_dataset = load_pkl(dataset_dir / 'dev.pkl')

    print('[*] Creating train/dev data loaders\n')
    if cfg.data_loader.batch_size % cfg.train.n_gradient_accumulation_steps != 0:
        print(
            '[!] n_gradient_accumulation_steps({}) is not a divider of '
            .format(cfg.train.n_gradient_accumulation_steps),
            'batch_size({})'.format(cfg.data_loader.batch_size))
        exit(1)
    cfg.data_loader.batch_size //= cfg.train.n_gradient_accumulation_steps
    train_data_loader = create_data_loader(
        train_dataset, word_vocab, char_vocab, **cfg.data_loader)
    dev_data_loader = create_data_loader(
        dev_dataset, word_vocab, char_vocab, **cfg.data_loader)
    
    print('[*] Creating model\n')
    t_total = int(len(train_dataset) / cfg.data_loader.batch_size / cfg.train.n_gradient_accumulation_steps) * cfg.train.n_epochs
    model = Model(device, cfg.net, cfg.optim, t_total)

    trainer = Trainer(
        cfg.net.bert_pretrained_model_name, device, cfg.train,
        train_data_loader, dev_data_loader, model,
        [CrossEntropyLoss(device, 'logits', 'label')], [Accuracy(device, 'label')],
        log_path, ckpt_dir)
    trainer.start()


if __name__ == "__main__":
    with ipdb.launch_ipdb_on_exception():
        sys.breakpointhook = ipdb.set_trace
        kwargs = parse_args()
        main(**kwargs)
