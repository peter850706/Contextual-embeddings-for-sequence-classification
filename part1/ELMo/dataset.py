import torch
from torch.utils.data import Dataset, DataLoader


class CorpusDataset(Dataset):
    """The own defined dataset class for training the ELMo.
    Args:
        data (): The data
        max_sequence_len (int): The length for padding a batch of samples to the same length.
        word_padding (int): The index for word padding.
        char_padding (int): The index for character padding.
    """
    def __init__(self, data, max_sequence_len=64, word_padding=0, char_padding=0):
        self.data = data
        self.max_sequence_len = max_sequence_len
        self.word_padding = word_padding
        self.char_padding = char_padding
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    def collate_fn(self, datas):
        """The own defined collate function for the dataloader.
        Args:
            datas (list): list of sentences.
        Returns:
            batch (dict):
                word_lens (LongTensor) (batch, ): The original length of the sequence (input are the same as target).
                forward (dict): The data for forward language model.
                    input (tensor) (batch, padded_len): The padded input sequence.
                    target (LongTensor) (batch, padded_len): The padded output sequence.
                backward (dict): the data for backward language model.
                    input (tensor) (batch, padded_len): The padded input sequence.
                    target (LongTensor) (batch, padded_len): The padded output sequence. 
        """
        batch = {'forward': {}, 'backward': {}}
                
        # build tensors of foward and backward input and target
        batch['word_lens'] = torch.tensor([min(len(data['forward']['input']), self.max_sequence_len)
                                           for data in datas], dtype=torch.long)
        padded_word_len = batch['word_lens'].max().item()
        
        char_lens = [len(word) 
                     for data in datas 
                     for word in data['forward']['input']]
        padded_char_len = max(char_lens)
        
        batch['forward']['input'] = torch.tensor([input_pad_to_len(data['forward']['input'], 
                                                                   padded_word_len, 
                                                                   padded_char_len,
                                                                   self.word_padding, 
                                                                   self.char_padding)
                                                  for data in datas], dtype=torch.long)
        batch['forward']['target'] = torch.tensor([target_pad_to_len(data['forward']['target'],
                                                                     padded_word_len, 
                                                                     self.word_padding)
                                                   for data in datas], dtype=torch.long)
        
        char_lens = [len(word) 
                     for data in datas 
                     for word in data['backward']['input']]
        padded_char_len = max(char_lens)
        
        batch['backward']['input'] = torch.tensor([input_pad_to_len(data['backward']['input'], 
                                                                    padded_word_len,  
                                                                    padded_char_len,
                                                                    self.word_padding, 
                                                                    self.char_padding)
                                                   for data in datas], dtype=torch.long)
        batch['backward']['target'] = torch.tensor([target_pad_to_len(data['backward']['target'], 
                                                                      padded_word_len, 
                                                                      self.word_padding)
                                                    for data in datas], dtype=torch.long)
        return batch

def input_pad_to_len(words, padded_word_len, padded_char_len, word_padding=0, char_padding=0):
    """Pad words to 'padded_word_len' with padding if 'len(words) < padded_word_len'. Then add padding so that each word has the same length 'padded_char_len'.
    Example:
        input_pad_to_len([[1, 2, 3],
                          [2, 3],
                          [3]], 5, -1, 0)
                          
        ==============>  [[1, 2, 3],
                          [2, 3, 0],
                          [3, 0, 0],
                          [-1, 0, 0],
                          [-1, 0, 0]]
    Args:
        words (list): List of list of the word index.
        padded_word_len (int): The length for padding a batch of sequences to the same length.
        padded_char_len (int): The length for padding each word in the sentences to the same length.
        word_padding (int): The index used to pad the input sequence.
        char_padding (int): The index used to pad each word in the input sequence.
    """
    if len(words) < padded_word_len:
        words += [[word_padding]] * (padded_word_len - len(words))
    words = [word + [char_padding] * (padded_char_len - len(word)) if len(word) < padded_char_len else word for word in words]
    return words
    
def target_pad_to_len(words, padded_word_len, word_padding=0):
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
    return words