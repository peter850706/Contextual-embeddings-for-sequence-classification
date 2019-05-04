import argparse
import logging
import os
import pdb
import sys
import traceback
import pickle
import random
from collections import Counter
from ELMo.processor import Processor

    
def main(args):
    if not os.path.exists(args.dest_dir):
        os.makedirs(args.dest_dir)
    processor = Processor()
    
    # collect the sentences appear in the data
    sentences_pkl_path = os.path.join(args.dest_dir, 'sentences.pkl')
    try:
        logging.info(f'Loading sentences from {sentences_pkl_path}.')
        with open(sentences_pkl_path, 'rb') as f:
            sentences = pickle.load(f)
    except:
        logging.info('Loading sentences Failed!! Reconstructing sentences.')
        logging.info('Collecting sentences from data/language_model/corpus_tokenized.txt.')
        with open('data/language_model/corpus_tokenized.txt', 'r') as f:
            sentences = f.readlines()
        sentences = sorted(list(set(sentences)))
        logging.info(f'The number of sentences in data/language_model/corpus_tokenized.txt is {len(sentences)}.')
        logging.info(f'Random select {args.num_sentences} sentences.')
        random.seed('corpus_tokenized.txt') # use a defined random seed
        sentences = random.sample(population=sentences, k=args.num_sentences)
        random.seed() # use the current system time as the random seed again
        logging.info(f'Saving sentences to {sentences_pkl_path}.')
        with open(sentences_pkl_path, 'wb') as f:
            pickle.dump(sentences, f)
            
    # collect the words from the sentences
    words_pkl_path = os.path.join(args.dest_dir, 'words.pkl')
    try:
        logging.info(f'Loading words from {words_pkl_path}.')
        with open(words_pkl_path, 'rb') as f:
            words = pickle.load(f)
    except:
        logging.info('Loading words Failed!! Reconstructing words.')
        logging.info('Collecting words from preprocessed sentences.')
        words = Counter(processor.collect_words(sentences, args.num_workers))
        logging.info(f'Saving words to {words_pkl_path}.')
        with open(words_pkl_path, 'wb') as f:
            pickle.dump(words, f)
            
    # collect the characters from the words
    chars_pkl_path = os.path.join(args.dest_dir, 'chars.pkl')
    try:
        logging.info(f'Loading chars from {chars_pkl_path}.')
        with open(chars_pkl_path, 'rb') as f:
            chars = pickle.load(f)
    except:
        logging.info('Loading chars Failed!! Reconstructing chars.')
        logging.info('Collecting chars from preprocessed words.')
        chars = Counter(processor.collect_chars(words, args.num_workers))
        logging.info(f'Saving chars to {chars_pkl_path}.')
        with open(chars_pkl_path, 'wb') as f:
            pickle.dump(chars, f)
        
    # filter the words by their frequency
    logging.info(f'The number of words before filtering is {len(words)}')
    min_word_frequency = 3
    for key, value in list(words.items()):
        if value < min_word_frequency:
            words.pop(key, None)
    logging.info(f'Filter the words that their frequency are less than {min_word_frequency}')
    logging.info(f'The number of words after filtering is {len(words)}')
    processor.update_word_vocabulary(words)
    
    # filter the characters by their frequency
    logging.info(f'The number of chars before filtering is {len(chars)}')
    min_char_frequency = 1000
    for key, value in list(chars.items()):
        if value < min_char_frequency:
            chars.pop(key, None)
    logging.info(f'Filter the characters that their frequency are less than {min_char_frequency}')
    logging.info(f'The number of chars after filtering is {len(chars)}')
    processor.update_char_vocabulary(chars)
    
    # save the character vocabulary
    char_vocabulary_pkl_path = os.path.join(args.dest_dir, 'char_vocabulary.pkl')
    logging.info(f'Saving character vocabulary to {char_vocabulary_pkl_path}.')
    with open(char_vocabulary_pkl_path, 'wb') as f:
        pickle.dump(processor.char_vocabulary, f)
    
    # pre-process the training and validation data
    logging.info('Random split the sentences into training and validation (0.99:0.01).')
    
    random.seed('training') # use a defined random seed
    train_data = random.sample(population=sentences, k=int(args.num_sentences * 0.99))
    random.seed() # use the current system time as the random seed again
    logging.info(f'The number of training sentences is {len(train_data)}.')
    train = processor.get_dataset(train_data, args.num_workers, args.max_sequence_len)
    logging.info(f'The number of training data is {len(train.data)}.')
    train_pkl_path = os.path.join(args.dest_dir, 'train.pkl')
    logging.info(f'Saving train to {train_pkl_path}')
    with open(train_pkl_path, 'wb') as f:
        pickle.dump(train, f)
        
    valid_data = sorted(list(set(sentences) - set(train_data)))
    logging.info(f'The number of validation sentences is {len(valid_data)}.')
    valid = processor.get_dataset(valid_data, args.num_workers, args.max_sequence_len)
    logging.info(f'The number of validation data is {len(valid.data)}.')
    valid_pkl_path = os.path.join(args.dest_dir, 'valid.pkl')
    logging.info(f'Saving valid to {valid_pkl_path}')
    with open(valid_pkl_path, 'wb') as f:
        pickle.dump(valid, f)        
    
def _parse_args():
    parser = argparse.ArgumentParser(description="Preprocess and generate preprocessed pickle.")
    parser.add_argument('dest_dir', type=str, help='[input] Path to the directory that .')
    parser.add_argument('--num_sentences', type=int, default=1000000, help='[input] The number of the selected sentences.')
    parser.add_argument('--num_workers', type=int, default=16)
    parser.add_argument('--max_sequence_len', type=int, default=64)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s | %(levelname)s | %(name)s: %(message)s', level=logging.INFO, datefmt='%m-%d %H:%M:%S')
    args = _parse_args()
    try:
        main(args)
    except KeyboardInterrupt:
        pass
    except BaseException:
        type, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)