import logging
from tqdm import tqdm
from multiprocessing import Pool
from ELMo.dataset import CorpusDataset


class Processor(object):
    def __init__(self):
        self.word_vocabulary = None
        self.char_vocabulary = None
        
    def split_sentence(self, sentence):
        """Split the sentence into words. The sentences in corpus_tokenized.txt are tokenized, so using str.split() and adding the special tokens <BOS> and <EOS>.
        Args:
            sentence (str): A sentence.
        Returns:
            words (list): A list of words.
        """
        words = ['<BOS>'] + sentence.split() + ['<EOS>']
        return words
    
    def split_word(self, word):
        """Split the word into characters.
        Args:
            word (str): A word.
        Returns:
            chars (list): A list of characters.
        """
        if word in ['<BOS>', '<EOS>']:
            chars = [word]
        else:
            chars = list(word)
        return chars
    
    def collect_words_worker(self, sentences):
        """The worker function for collect_words function.
        Args:
            sentences (list): A list of sentences.
        Returns:
            processed (list): A list of processed samples.
        """
        processed = []
        for sentence in tqdm(sentences):
            processed.extend(self.split_sentence(sentence))
        return processed
    
    def collect_chars_worker(self, words):
        """The worker function for collect_chars function.
        Args:
            words (list): A list of tuples (word, frequency).
        Returns:
            processed (list): A list of processed samples.
        """
        processed = []
        for word, frequency in tqdm(words):
            processed.extend(self.split_word(word) * frequency)
        return processed
    
    def collect_words(self, sentences, num_workers):
        """Collect the words from the sentences.
        Args:
            sentences (list): A list of sentences.
            num_workers (int): The number of the processes.
        Returns:
            word (list): A list of repeatable words (a word could appear several times).
        """
        results = [None] * num_workers
        with Pool(processes=num_workers) as pool:
            for i in range(num_workers):
                batch_start = (len(sentences) // num_workers) * i                
                batch_end = len(sentences) if i == num_workers - 1 else (len(sentences) // num_workers) * (i + 1)
                bacth = sentences[batch_start: batch_end]
                results[i] = pool.apply_async(self.collect_words_worker, [bacth])
            pool.close()
            pool.join()
            
        words = []    
        for result in results:
            words += result.get()
        return words
    
    def collect_chars(self, words, num_workers):
        """Collect the characters from the words.
        Args:
            words (Counter): A dict of words and their corresponding frequency.
            num_workers (int): The number of the processes.
        Returns:
            chars (list): A list of repeatable characters (a character could appear several times).
        """ 
        results = [None] * num_workers
        with Pool(processes=num_workers) as pool:
            for i in range(num_workers):
                batch_start = (len(words) // num_workers) * i
                batch_end = len(words) if i == num_workers - 1 else (len(words) // num_workers) * (i + 1)
                bacth = list(words.items())[batch_start: batch_end]
                results[i] = pool.apply_async(self.collect_chars_worker, [bacth])
            pool.close()
            pool.join()
            
        chars = []
        for result in results:
            chars += result.get()
        return chars
    
    def update_word_vocabulary(self, words):
        """Update the word vocabulary in descending order according to the words' frequency.
        Args:
            words (Counter): A dict of words and their corresponding frequency.
        """
        self.word_vocabulary = dict({'<pad>': 0, '<unk>': 1})
        words = words.most_common()
        self.word_vocabulary.update(dict([(key, i + len(self.word_vocabulary)) for i, (key, _) in enumerate(words)]))
    
    def update_char_vocabulary(self, chars):
        """Update the character vocabulary in descending order according to the characters' frequency.
        Args:
            chars (Counter): A dict of characters and their corresponding frequency.
        """
        self.char_vocabulary = dict({'<pad>': 0, '<unk>': 1})
        chars = chars.most_common()
        self.char_vocabulary.update(dict([(key, i + len(self.char_vocabulary)) for i, (key, _) in enumerate(chars)]))
        
    def process_samples(self, sentence, max_sequence_len=64):
        """Split the sentence into several samples for every max_sequence_len (64) words.
        Args:
            sentence (str): The original sentence in the data.
            max_sequence_len (int): The max length of a sample.
        Returns:
            processed_samples (list): A list of samples.
        """
        processed_samples = []
        sentence = self.split_sentence(sentence)
        step = max_sequence_len + 1
        for i in range(0, len(sentence), step):
            processed_sample = {}
            sample = sentence[i : i + step] if i + step <= len(sentence) else sentence[i:]
            
            # if the length of the sample ############################
            if len(sample) < 2:
                break
                
            input, target = [], []
            for word in sample:
                chars = self.split_word(word)
                input.append([self.char_vocabulary.get(char, self.char_vocabulary['<unk>']) for char in chars])
            target = [self.word_vocabulary.get(word, self.word_vocabulary['<unk>']) for word in sample]
            
            processed_sample['forward'] = {'input': input[:-1], 'target': target[1:]} # forward
            input.reverse()
            target.reverse()
            processed_sample['backward'] = {'input': input[:-1], 'target': target[1:]} # backward
            processed_samples.append(processed_sample)
        return processed_samples
    
    def get_dataset_worker(self, sentences, max_sequence_len=64):
        """The worker function for get_dataset function.
        Args:
            sentences (list): A list of sentences.
        Returns:
            processed_samples (list): A list of processed samples.
        """
        processed_samples = []
        for sentence in tqdm(sentences):
            processed_samples.extend(self.process_samples(sentence, max_sequence_len))
        return processed_samples
    
    def get_dataset(self, sentences, num_workers, max_sequence_len=64):
        """Load sentences and create the CorpusDataset objects for training and validating.
        Args:
            sentences (list): A list of sentences.
        Returns:
            dataset (CorpusDataset): The CorpusDataset
        """
        assert self.word_vocabulary is not None and self.char_vocabulary is not None
        
        results = [None] * num_workers
        with Pool(processes=num_workers) as pool:
            for i in range(num_workers):
                batch_start = (len(sentences) // num_workers) * i
                batch_end = len(sentences) if i == num_workers - 1 else (len(sentences) // num_workers) * (i + 1)
                batch = sentences[batch_start: batch_end]
                results[i] = pool.apply_async(self.get_dataset_worker, [batch], kwds={'max_sequence_len': max_sequence_len})
            pool.close()
            pool.join()
            
        data = []
        for result in results:
            data += result.get()
        
        dataset = CorpusDataset(data=data, 
                                max_sequence_len=max_sequence_len,
                                word_padding=self.word_vocabulary['<pad>'], 
                                char_padding=self.char_vocabulary['<pad>'])
        return dataset