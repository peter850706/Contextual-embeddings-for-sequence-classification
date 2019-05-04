import argparse
import csv
import pickle
import re
import string
import sys
from collections import Counter
from pathlib import Path

import ipdb
import spacy
from box import Box
from tqdm import tqdm

from BERT.dataset import Part2Dataset


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('test_csv_path', type=Path, help='Test .csv file path')
    parser.add_argument('dataset_dir', type=Path, help='Target dataset directory')
    args = parser.parse_args()
    return vars(args)

def load_data(mode, data_path, nlp):
    print('[*] Loading {} data from {}'.format(mode, data_path))
    with data_path.open() as f:
        reader = csv.DictReader(f)
        data = [r for r in reader]

    for d in tqdm(data, desc='[*] Tokenizing', dynamic_ncols=True):
        text = re.sub('-+', ' ', d['text'])
        text = re.sub('\s+', ' ', text)
        doc = nlp(text)
        d['text'] = [token.text for token in doc]
    print('[-] {} data loaded\n'.format(mode.capitalize()))
    return data

def create_dataset(data, dataset_dir):
    for m, d in data.items():
        print('[*] Creating {} dataset'.format(m))
        dataset = Part2Dataset(d)
        dataset_path = (dataset_dir / '{}.pkl'.format(m))
        with dataset_path.open(mode='wb') as f:
            pickle.dump(dataset, f)
        print('[-] {} dataset saved to {}\n'.format(m.capitalize(), dataset_path))


def main(test_csv_path, dataset_dir):    
    print('[-] Test dataset will be saved to {}\n'.format(dataset_dir))

    output_files = ['test.pkl']
    if any([(dataset_dir / p).exists() for p in output_files]):
        print('[!] Directory already contains test dataset')
        exit(1)

    nlp = spacy.load('en')
    nlp.disable_pipes(*nlp.pipe_names)

    data = {'test': load_data('test', test_csv_path, nlp)}
    create_dataset(data, dataset_dir)


if __name__ == "__main__":
    with ipdb.launch_ipdb_on_exception():
        sys.breakpointhook = ipdb.set_trace
        kwargs = parse_args()
        main(**kwargs)
